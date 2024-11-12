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

seed = 37
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


transform = transforms.Compose(
    [
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Set the simulation rounds


num_clients = 24


num_rounds = 2000
batch_size = 256    #1024
accumulation_steps = 1
total_samples = len(train_dataset)
fraction = 0.1  # 0.2
learning_rate = 0.01  # 0.01
lr_decay = 0.97  # 0.97
lr_decay_step = 40 # 25


num_samples = int(total_samples * fraction)

# Generate random indices for the subset
indices = list(range(total_samples))
# random.shuffle(indices)
# subset_indices = indices[:num_samples]

total_indices = list(range(total_samples))
# random.shuffle(total_indices)
# test_dataset_full = Subset(train_dataset, total_indices[:1000])


total_sample_num = num_samples*num_clients
client_indices = list()
# for i in range(num_clients):
subset_indices = [list() for _ in range(num_clients)]

iter_num = total_sample_num//total_samples

num_nodes_index = [_ for _ in range(num_clients)]

for _ in range(iter_num):
    random.shuffle(total_indices)
    cur_samples = total_indices[:total_sample_num]
    cur_samples = np.array_split(cur_samples, num_clients)
    random.shuffle(num_nodes_index)
    for j in num_nodes_index:
        subset_indices[j].extend(cur_samples[j])

if total_sample_num % total_samples > 0:
    random.shuffle(total_indices)
    cur_samples = total_indices[:total_sample_num % total_samples]
    cur_samples = np.array_split(cur_samples, num_clients)
    random.shuffle(num_nodes_index)
    for j in num_nodes_index:
        subset_indices[j].extend(cur_samples[j])

client_indices = subset_indices



def customize_topology():
    topology = list()
    for i in range(num_clients):
        topology.append([
                        # (i-2+num_clients) % num_clients,
                         (i-1+num_clients) % num_clients,
                         (i+1+num_clients) % num_clients,
                        # (i+2+num_clients) % num_clients
                        ])
    return topology


def create_iid_splits(_, indices, num_clients):
    node_indices = [list() for _ in range(num_clients)]
    for i, indice in enumerate(indices):
        # x = np.random.randint(0, 4)
        for id in indice:
            node_indices[i].append(id)
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

    # Extract labels for the given indices
    labels = [int(dataset.targets[idx]) for idx in indices]

    # Count occurrences of each class label
    label_counts = Counter(labels)

    total_count = sum(label_counts.values())
    print_str = f"\nClass distribution in {title}:\nTotal_count {total_count}\n"
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        ratio = count / total_count
        print_str = print_str + \
            f"  Class {label}: Count {count}, Ratio {ratio:.4f}\n"
    print(print_str)


def generate_doubly_stochastic_matrix(n, tol=1e-10, max_iter=1000):
    """
    Generates an n x n doubly stochastic matrix using the Sinkhorn-Knopp algorithm.

    Parameters:
    - n (int): Size of the matrix.
    - tol (float): Tolerance for convergence.
    - max_iter (int): Maximum number of iterations.

    Returns:
    - A (ndarray): An n x n doubly stochastic matrix.
    """
    # Start with a random positive matrix
    A = np.random.rand(n, n)

    # Iteratively normalize rows and columns
    for _ in range(max_iter):
        # Normalize rows
        A /= A.sum(axis=1, keepdims=True)
        # Normalize columns
        A /= A.sum(axis=0, keepdims=True)
        # Check convergence
        if np.allclose(A.sum(axis=1), 1, atol=tol) and np.allclose(A.sum(axis=0), 1, atol=tol):
            break
    else:
        print("Warning: Maximum iterations reached before convergence.")
    return A


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, hidden_size=200, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Client(threading.Thread):
    """
    Represents a client that manages its nodes and communicates with other clients.
    """

    def __init__(self, client_id, node_indices_list, dataset, num_clients, clients_list, joint_clients, learn_rate, global_model=None):
        threading.Thread.__init__(self)
        self.client_id = client_id
        # self.nodes = []
        self.gradients = None
        self.dataset = dataset
        # Initialize global model if not provided
        # self.global_model = global_model if global_model else SimpleCNN()
        # self.global_model = global_model if global_model else ComplexCNN(
        #     input_channel=1)
        self.global_model = global_model
        self.num_clients = num_clients
        # self.received_models = []  # List to store models received from other clients
        self.received_models = Queue()
        self.received_models_q = Queue()
        self.aggregator_queue = Queue()  # Queue to collect gradients from nodes
        self.clients_list = clients_list  # Reference to other clients
        self.joint_clients = joint_clients
        self.learn_rate = learn_rate
        # self.optimizer = torch.optim.Adam(
        #     self.global_model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.global_model.parameters(), lr=self.learn_rate)
        self.data_loader = DataLoader(
            Subset(dataset, node_indices_list),
            batch_size=batch_size,
            shuffle=True
        )

        # print_class_distribution(
        #     dataset, node_indices_list, title=f"Client {client_id}")

    def run(self):
        print(f"Client {self.client_id} starting.")
        # Start all nodes under this client
        # for node in self.nodes:
        #     node.start()
        # Collect gradients from nodes
        collected_gradients = []
        # for _ in self.nodes:
        #     gradients = self.aggregator_queue.get()
        #     collected_gradients.append(gradients)
        # # Wait for all nodes to complete
        # for node in self.nodes:
        #     node.join()
        self.global_model.train()
        # self.global_model.zero_grad()
        t = 0
        for data, target in self.data_loader:
            data = data.cuda()
            target = target.cuda()
            if t == 0:
                self.global_model.zero_grad()  # lulu
            output = self.global_model(data)
            loss = self.criterion(output, target)
            # self.optimizer.zero_grad()
            loss = loss/accumulation_steps
            loss.backward()
            if t+1 == accumulation_steps:
                break
            t = t+1
            # self.optimizer.step()
        # Extract gradients
        self.gradients = [
            param.grad.clone().detach().cpu() for param in self.global_model.cpu().parameters()
        ]
        # self.gradients = [param.grad.clone()
        #                   for param in self.global_model.cpu().parameters()]
        # # Send gradients to aggregator
        # self.aggregator_queue.put(self.gradients)
        # print(f"Client {self.client_id} collected all gradients.")
        # Aggregate gradients and update the global model
        self.aggregate_and_update(collected_gradients)
        print(f"Client {self.client_id} has updated the global model.")
        # Communicate updated model parameters to other clients
        self.communicate()
        print(f"Client {self.client_id} finished communication.")
        # Average the received models to update the local model
        self.average_models()
        print(
            f"Client {self.client_id} updated its model by averaging received models.")

    def aggregate_and_update(self, collected_gradients):
        """
        Aggregates gradients from all nodes and updates the global model.
        """
        # # Initialize aggregated gradients
        # aggregated_gradients = [torch.zeros_like(
        #     param) for param in self.global_model.parameters()]
        # num_nodes = len(collected_gradients)
        # # Sum gradients from all nodes
        # for gradients in collected_gradients:
        #     for idx, grad in enumerate(gradients):
        #         aggregated_gradients[idx] += grad
        # # Average the gradients
        # aggregated_gradients = [
        #     grad / num_nodes for grad in aggregated_gradients]
        # # Update global model parameters
        # self.global_model.train()
        # # with torch.no_grad():
        # for param, grad in zip(self.global_model.parameters(), aggregated_gradients):
        #     # param -= 0.001 * grad  # Update rule with learning rate 0.01
        #     param.grad = grad
        # self.optimizer.step()
        pass

    def communicate(self):
        """
        Sends the updated model parameters to all other clients.
        """
        for client_id in self.joint_clients:
            # if client_id != self.client_id:
            # print("communicate, self.id, client id",
            #       self.client_id, client_id, self.joint_clients)
            # if (client_id in self.joint_clients) and (client_id != self.client_id):
            target_client = self.clients_list[client_id]
            # Send (copy) the global model parameters to the target client
            # print(
            #     f"client {self.client_id} send model para to target client {target_client.client_id}")
            # parameters_lock[client_id].acquire()
            # print("++++++++")
            # target_client.received_models.append(
            #     copy.deepcopy(self.global_model))
            # parameters_lock[client_id].release()
            target_client.received_models_q.put(
                copy.deepcopy(self.global_model))
            print(
                f"Client {self.client_id} sent model parameters to Client {client_id}.")

            if self.received_models_q.qsize() != 0:
                tmp_received_models = []
                for _ in range(self.received_models_q.qsize()):
                    tmp_model = self.received_models_q.get()
                    tmp_received_models.append(tmp_model)
                    self.received_models.put(tmp_model)
                tmp_received_models.append(self.global_model)
                state_dicts = [model.state_dict()
                               for model in tmp_received_models]
                # Get keys from the state_dict
                param_keys = state_dicts[0].keys()
                # Initialize new state_dict for averaged parameters
                averaged_state_dict = {}
                for key in param_keys:
                    # Sum parameters from all models
                    params = [state_dict[key]
                              for state_dict in state_dicts]
                    # Stack parameters and compute mean
                    stacked_params = torch.stack(params, dim=0)
                    averaged_param = torch.mean(stacked_params, dim=0)
                    averaged_state_dict[key] = averaged_param
                # Load averaged parameters into the global model
                self.global_model.load_state_dict(averaged_state_dict)

    def average_models(self):
        """
        Averages the received models to update the local global model.
        """
        while True:
            # time.sleep(0.1)
            if self.received_models.qsize()+self.received_models_q.qsize() == len(self.joint_clients):
                break
            # parameters_lock[self.client_id].acquire()
            # print("client, received, joint", self.client_id, len(
            #     self.received_models), len(self.joint_clients))
            # if len(self.received_models) == len(self.joint_clients):
            #     break
            # parameters_lock[self.client_id].release()
        # num_models = len(self.received_models)
        if self.received_models.qsize() == len(self.joint_clients):
            self.global_model.train()
            for param, grad in zip(self.global_model.parameters(), self.gradients):
                # param -= 0.001 * grad  # Update rule with learning rate 0.01
                param.grad = grad
            self.optimizer.step()
        else:
            for _ in range(self.received_models_q.qsize()):
                self.received_models.put(self.received_models_q.get())
            print(
                f"Client {self.client_id} received {self.received_models.qsize()} model parameters.")
            # # if not self.received_models:
            # if num_models == 0:
            #     return  # No models received
            # Include own model in averaging
            self.received_models.put(self.global_model)
            # Average parameters using state_dict
            # Collect state_dicts from all models
            state_dicts = [self.received_models.get().state_dict()
                           for i in range(self.received_models.qsize())]
            # Get keys from the state_dict
            param_keys = state_dicts[0].keys()
            # Initialize new state_dict for averaged parameters
            averaged_state_dict = {}
            for key in param_keys:
                # Sum parameters from all models
                params = [state_dict[key] for state_dict in state_dicts]
                # Stack parameters and compute mean
                stacked_params = torch.stack(params, dim=0)
                averaged_param = torch.mean(stacked_params, dim=0)
                averaged_state_dict[key] = averaged_param
            # Load averaged parameters into the global model
            self.global_model.load_state_dict(averaged_state_dict)
            # self.optimizer.zero_grad()
            # Clear received models for the next round
            self.received_models = Queue()
            self.global_model.train()
            for param, grad in zip(self.global_model.parameters(), self.gradients):
                # param -= 0.001 * grad  # Update rule with learning rate 0.01
                param.grad = grad
            self.optimizer.step()

# Define test function


def test_global_model(global_model, test_loader):
    global_model = global_model.cuda()
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


# Number of clients
# num_clients = 24

if __name__ == "__main__":
    f = open(
        "./results/res_adpsgd_iid_{}_{}.txt".format(num_clients, num_rounds), "a+")
    # Initialize clients' global models (None at the start)
    # clients_global_models = [ComplexCNN().cuda() for _ in range(num_clients)]
    init_model = SimpleCNN()
    clients_global_models = [copy.deepcopy(
        init_model).cuda() for _ in range(num_clients)]
    joint_clients = customize_topology()
    # Lists to store accuracy and loss trends
    accuracy_list = []
    loss_list = []
    # Prepare test loader
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    best_acc, best_round = 0.0, 0
    accuracy_list = []
    loss_list = []
    node_indices_list = create_iid_splits(
        train_dataset, client_indices, num_clients)
    for round_num in range(num_rounds):
        start_time = time.time()
        print(f"\n=== Round {round_num + 1} ===")
        # Create a list to hold clients for this round
        clients_list = []
        # Create clients and their nodes for this round
        for i, _ in enumerate(client_indices):
            # node_indices_list = create_noniid_splits(train_dataset, c_indices, 4)
            # Use the previous global_model if exists
            global_model = clients_global_models[i]
            client = Client(i, node_indices_list[i], train_dataset,
                            num_clients, clients_list, joint_clients[i],
                            learning_rate, global_model=global_model)
            clients_list.append(client)
        # Update clients_list in each client
        for client in clients_list:
            client.clients_list = clients_list
        # Start all clients
        for client in clients_list:
            client.start()
        # Wait for all clients to complete
        for client in clients_list:
            client.join()
        # Store the updated global models for the next round
        # clients_global_models = [client.global_model for client in clients_list]
        # Optionally, you can evaluate the global model here
        # For example, test accuracy on a validation set
        # Store the updated global models for the next round
        clients_global_models = [
            copy.deepcopy(client.global_model).cuda() for client in clients_list]
        clients_list.clear()
        torch.cuda.empty_cache()
        if (round_num+1) % lr_decay_step == 0:
            learning_rate = learning_rate * lr_decay
        
        # Evaluate the global model (using the first client's model)
        global_model = clients_global_models[0]
        # accuracy, avg_loss = test_global_model(global_model, test_loader)
        # accuracy_list.append(accuracy)
        # loss_list.append(avg_loss)
        # global_model = clients_global_models[0]
        accuracy, avg_loss = test_global_model(global_model, test_loader)
        accuracy_list.append(accuracy)
        loss_list.append(avg_loss)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_round = round_num
        end_time = time.time()
        print(
            f"Round {round_num + 1}: Test Accuracy: {accuracy*100:.2f}%, Test Loss: {avg_loss:.4f}, Learning rate: {learning_rate:.4f}"
        )
        f.write("round {:05}, acc {:.6f}, loss {:.6f}, best acc {:.6f}, best round {:05}, time {:.3f}\n".format(
                round_num, accuracy, avg_loss, best_acc, best_round, end_time-start_time))
        f.flush()
