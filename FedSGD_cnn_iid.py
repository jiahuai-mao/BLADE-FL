import threading
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import copy
import time
from queue import Queue

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

num_clients = 1
num_nodes_list = [24]

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)


num_rounds = 2000
batch_size = 2048
accumulation_steps = 1
total_samples = len(train_dataset)
fraction = 0.4
num_samples = int(total_samples * fraction)
total_indices = list(range(total_samples))

client_indices = []
for i in range(num_clients):
    subset_indices = []
    for _ in range(num_nodes_list[i]):
        random.shuffle(total_indices)
        subset_indices.extend(total_indices[:num_samples])
    client_indices.append(subset_indices)


test_dataset_full = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)


def create_iid_splits(dataset, indices, num_nodes_list):
    node_indices = list()

    for i, num_nodes in enumerate(num_nodes_list):

        cur_indices = indices[i]
        node_indice = [list() for _ in range(num_nodes)]
        for j in range(num_nodes):
            random.shuffle(cur_indices)
            subset_indices = cur_indices[:num_samples]
            node_indice[j].extend(subset_indices)
        node_indices.append(node_indice)

    return node_indices


def print_class_distribution(dataset, indices, title="Dataset"):
    """
    Prints the class distribution (counts and ratios) for a given set of indices.

    Args:
        dataset: The full dataset (e.g., train_dataset).
        indices: A list of indices representing the subdataset.
        title: A string title to identify the subdataset.
    """
    from collections import Counter

    labels = [int(dataset.targets[idx]) for idx in indices]

    label_counts = Counter(labels)

    total_count = sum(label_counts.values())
    print_str = f"\nClass distribution in {title}:\nTotal_count {total_count}\n"
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        ratio = count / total_count
        print_str = print_str + \
            f"  Class {label}: Count {count}, Ratio {ratio:.4f}\n"
    print(print_str)


class ComplexCNN(nn.Module):
    def __init__(self, input_channel=1, num_classes=10):
        super(ComplexCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Tanh(),

        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.Tanh(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x


class Node(threading.Thread):
    """
    Represents a node in the network.
    Each node performs local training and computes gradients.
    """

    def __init__(self, node_id, data_indices, dataset, global_model, aggregator_queue):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.data_loader = DataLoader(
            Subset(dataset, data_indices), batch_size=batch_size, shuffle=True
        )
        self.local_model = copy.deepcopy(global_model).cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.gradients = None

        self.aggregator_queue = aggregator_queue

    def run(self):
        print(f"{self.node_id} starting local training.")

        self.local_model.train()

        t = 0
        for data, target in self.data_loader:
            data = data.cuda()
            target = target.cuda()

            if t == 0:
                self.local_model.zero_grad()

            output = self.local_model(data)
            loss = self.criterion(output, target)
            loss = loss / accumulation_steps
            loss.backward()

            if (t + 1) % accumulation_steps == 0:
                break
            t += 1
        self.gradients = [
            param.grad.clone().detach().cpu() for param in self.local_model.parameters()
        ]
        self.aggregator_queue.put(self.gradients)
        print(f"{self.node_id} sent gradients to aggregator.")


class Client(threading.Thread):
    """
    Represents a client that manages its nodes and communicates with other clients.
    """

    def __init__(
        self,
        client_id,
        node_indices_list,
        dataset,
        num_clients,
        clients_list,
        joint_clients,
        global_model=None,
    ):
        threading.Thread.__init__(self)
        self.client_id = client_id
        self.nodes = []
        self.dataset = dataset

        self.global_model = global_model
        self.num_clients = num_clients

        self.received_models = Queue()
        self.received_models_q = Queue()
        self.aggregator_queue = Queue()
        self.clients_list = clients_list
        self.joint_clients = joint_clients

        self.optimizer = torch.optim.SGD(
            self.global_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )

        for i, node_indices in enumerate(node_indices_list):

            node = Node(
                f"Client{client_id}_Node{i}",
                node_indices,
                dataset,
                self.global_model,
                self.aggregator_queue,
            )
            self.nodes.append(node)

    def run(self):
        print(f"Client {self.client_id} starting.")

        for node in self.nodes:
            node.start()

        for node in self.nodes:
            node.join()

        collected_gradients = []
        for _ in self.nodes:
            gradients = self.aggregator_queue.get()
            collected_gradients.append(gradients)
        print(f"Client {self.client_id} collected all gradients.")

        self.aggregate_and_update(collected_gradients)
        print(f"Client {self.client_id} has updated the global model.")

        self.communicate()
        print(f"Client {self.client_id} finished communication.")

        self.average_models()
        print(
            f"Client {self.client_id} updated its model by averaging received models."
        )

    def aggregate_and_update(self, collected_gradients):
        """
        Aggregates gradients from all nodes and updates the global model.
        """

        aggregated_gradients = [
            torch.zeros_like(param) for param in self.global_model.parameters()
        ]
        num_nodes = len(collected_gradients)

        for gradients in collected_gradients:
            for idx, grad in enumerate(gradients):
                aggregated_gradients[idx] += grad.cuda()

        self.aggregated_gradients = [
            grad / num_nodes for grad in aggregated_gradients]

    def communicate(self):
        """
        Sends the updated model parameters to all other clients.
        """
        pass

    def average_models(self):
        """
        Averages the received models to update the local global model.
        """
        self.global_model.train()
        for param, grad in zip(
            self.global_model.parameters(), self.aggregated_gradients
        ):

            param.grad = grad
        self.optimizer.step()
        print(f"client {self.client_id}, averge model complete.")


def test_global_model(global_model, test_loader):
    global_model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    average_loss = total_loss / total
    return accuracy, average_loss


if __name__ == "__main__":
    f = open("./results/res_FedSGD_iid_{}_{}_{}.txt".format(num_clients, num_rounds,
             "-".join([str(i) for i in num_nodes_list])), "a+")
    init_model = ComplexCNN()
    clients_global_models = [copy.deepcopy(
        init_model).cuda() for _ in range(num_clients)]
    test_loader = DataLoader(
        test_dataset_full, batch_size=batch_size, shuffle=False)

    best_acc, best_round = 0.0, 0
    accuracy_list = []
    loss_list = []

    node_indices_list = create_iid_splits(
        train_dataset, client_indices, num_nodes_list
    )
    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")

        clients_list = []

        for i in range(num_clients):

            global_model = clients_global_models[i]
            client = Client(
                i,
                node_indices_list[i],
                train_dataset,
                num_clients,
                clients_list,
                None,
                global_model=global_model,
            )
            clients_list.append(client)
        for client in clients_list:
            client.clients_list = clients_list
        for client in clients_list:
            client.start()
        for client in clients_list:
            client.join()
        clients_global_models = [
            client.global_model for client in clients_list]

        global_model = clients_global_models[0]
        accuracy, avg_loss = test_global_model(global_model, test_loader)
        accuracy_list.append(accuracy)
        loss_list.append(avg_loss)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_round = round_num
        print(
            f"Round {round_num + 1}: Test Accuracy: {accuracy*100:.2f}%, Test Loss: {avg_loss:.4f}"
        )
        f.write("round {:05}, acc {:.6f}, loss {:.6f}, best acc {:.6f}, best round {:05}\n".format(
                round_num, accuracy, avg_loss, best_acc, best_round))
        f.flush()
