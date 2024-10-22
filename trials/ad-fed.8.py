import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import copy
import time
from queue import Queue
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# num_clients = 1
num_clients = 4
# num_nodes_list = [6, 6, 6, 6]
num_nodes_list = [2, 2, 2, 2]
# num_nodes_list = [1]

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)


# Run the simulation for 10 rounds
num_rounds = 30000
batch_size = 512
total_samples = len(train_dataset)
fraction = 0.26  # Change to 0.3 for 30%
num_samples = int(total_samples * fraction)
total_indices = list(range(total_samples))

client_indices = []
for i in range(num_clients):
    subset_indices = total_indices[i*num_samples:(i+1)*num_samples]
    client_indices.append(subset_indices)

# for i in range(num_clients):
#     subset_indices = total_indices[i*num_samples:(i+1)*num_samples]
#     client_indices.append(subset_indices)

random.shuffle(total_indices)
# test_dataset_full = torchvision.datasets.FashionMNIST(
#     root="./data", train=False, download=True, transform=transform
# )
test_dataset_full = Subset(
    train_dataset, total_indices[:int(total_samples*fraction)])


def customize_topology():
    topology = list()
    for i in range(num_clients):
        topology.append(
            [(i - 1 + num_clients) % num_clients,
             (i + 1 + num_clients) % num_clients]
        )
    return topology


def create_iid_splits(dataset, indices, num_nodes):
    node_indices = list()
    for _ in range(num_nodes):
        random.shuffle(indices)
        subset_indices = indices[:batch_size*2]
        node_indices.append(subset_indices)
    return node_indices


def create_noniid_splits(dataset, indices, num_nodes):
    """
    Splits the data among nodes in a non-IID manner.
    Each node gets data from certain classes only.
    """
    labels = np.array(dataset.targets)[indices]
    classes = np.unique(labels)
    np.random.shuffle(classes)
    class_splits = np.array_split(classes, num_nodes)
    node_indices = []
    for class_split in class_splits:
        idx = [i for i, label in zip(indices, labels) if label in class_split]
        node_indices.append(idx)
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
    print_str = f"\nClass distribution in {title}:\n"
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
            # nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.Tanh(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print("111 ",x.shape)
        x = torch.flatten(x, 1)
        # print("222 ",x.shape)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    def __init__(self, input_channel=1, output_size=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 4, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh()
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * 32, 1024),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(1024, output_size)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x = x.reshape([-1, 1, 7, 8])
        # print("x = ", x.shape)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print("x.view = ", x.shape, "linear = ", 32*32*32)
        x = self.classifier(x)
        return x


class Node(threading.Thread):
    """
    Represents a node in the network.
    Each node performs local training and computes gradients.
    """

    def __init__(self, node_id, data_indices, dataset, global_model, aggregator_queue):
        threading.Thread.__init__(self)
        # print(batch_size, len(data_indices))
        self.node_id = node_id
        self.data_loader = DataLoader(
            Subset(dataset, data_indices), batch_size=batch_size, shuffle=True
        )
        self.local_model = copy.deepcopy(global_model).cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.gradients = None  # Placeholder for storing gradients
        # Queue to send gradients to aggregator
        self.aggregator_queue = aggregator_queue

    def run(self):
        print(f"{self.node_id} starting local training.")
        # Perform forward and backward passes to compute gradients
        # self.local_model = self.local_model.cuda()
        self.local_model.train()
        # self.local_model.zero_grad()
        gradient_list = []
        # iter = 0
        # Hyperparameters for Adam optimizer
        lr = 0.001       # Learning rate
        beta1 = 0.9     # Exponential decay rate for the first moment estimates
        beta2 = 0.999   # Exponential decay rate for the second moment estimates
        epsilon = 1e-8  # Small value to prevent division by zero

        # Initialize moment vectors and time step
        m = []
        v = []
        t = 0  # Time step
        for name, param in enumerate(self.local_model.parameters()):
            # m[name] = torch.zeros_like(param)
            # v[name] = torch.zeros_like(param)
            m.append(0)
            v.append(0)
            gradient_list.append(0)
        for data, target in self.data_loader:
            data = data.cuda()
            target = target.cuda()
            output = self.local_model(data)
            loss = self.criterion(output, target)

            # self.local_model.zero_grad()
            loss.backward()
            t += 1
            tmp = []
            # for name, param in enumerate(self.local_model.parameters()):
            # if param.grad is not None:
            #     grad = param.grad.clone().detach().cpu()

            #     # Update first moment estimate
            #     m[name] = beta1 * m[name] + (1 - beta1) * grad

            #     # Update second moment estimate
            #     v[name] = beta2 * v[name] + (1 - beta2) * grad * grad

            #     # Compute bias-corrected first moment estimate
            #     m_hat = m[name] / (1 - beta1 ** t)

            #     # Compute bias-corrected second moment estimate
            #     v_hat = v[name] / (1 - beta2 ** t)

            #     # Update parameters
            #     tmp.append(
            #         m_hat / (torch.sqrt(v_hat) + epsilon))

            # gradient_list.append(tmp)
            # iter += 1
            # if len(gradient_list) == 0:
            #     gradient_list = [
            #         grad for grad in tmp
            #     ]
            # else:
            #     gradient_list = [
            #         grad1 + grad2
            #         for grad1, grad2 in zip(gradient_list, tmp)
            #     ]
            break
            # break  # For demonstration, we only process one batch
        # Extract gradients
        # self.gradients = [grad / t for grad in gradient_list]
        # self.gradients = [grad.cuda() for grad in self.gradients]
        self.gradients = [
            param.grad.clone().detach().cpu() for param in self.local_model.cpu().parameters()
        ]
        # self.gradients = [1.0*grad/len(gradient_list)
        #                   for grad in gradient_list]
        # Send gradients to aggregator
        self.aggregator_queue.put(self.gradients)
        print(f"{self.node_id} sent gradients to aggregator.")
        del (self.local_model)
# I have to check the grads in each batch is same or not. In other words,
# the grads that were put in aggregator_queue are the final batch-generated or with full iteration.


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
        # Initialize global model if not provided\
        self.global_model = global_model
        self.num_clients = num_clients
        # self.received_models = []  # List to store models received from other clients
        self.received_models = Queue()
        self.received_models_q = Queue()
        self.aggregator_queue = Queue()  # Queue to collect gradients from nodes
        self.clients_list = clients_list  # Reference to other clients
        self.joint_clients = joint_clients
        # self.optimizer = torch.optim.Adam(
        #     self.global_model.parameters(), lr=0.001)
        self.optimizer = torch.optim.SGD(
            self.global_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )

        # print(f"client{self.client_id}, joint_clients: {self.joint_clients}")
        # Initialize nodes
        for i, node_indices in enumerate(node_indices_list):
            # print_class_distribution(
            #     dataset, node_indices, title=f"Client {client_id} Node {i}")
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
        # Start all nodes under this client
        for node in self.nodes:
            node.start()
        # Collect gradients from nodes
        collected_gradients = []
        for _ in self.nodes:
            gradients = self.aggregator_queue.get()
            collected_gradients.append(gradients)
        # Wait for all nodes to complete
        for node in self.nodes:
            node.join()
        print(f"Client {self.client_id} collected all gradients.")
        # Aggregate gradients and update the global model
        self.aggregate_and_update(collected_gradients)
        print(f"Client {self.client_id} has updated the global model.")
        # Communicate updated model parameters to other clients
        self.communicate()
        print(f"Client {self.client_id} finished communication.")
        # Average the received models to update the local model
        self.average_models()
        print(
            f"Client {self.client_id} updated its model by averaging received models."
        )

    def aggregate_and_update(self, collected_gradients):
        """
        Aggregates gradients from all nodes and updates the global model.
        """
        # Initialize aggregated gradients
        # aggregated_gradients = [
        #     torch.zeros_like(param) for param in self.global_model.parameters()
        # ]
        # num_nodes = len(collected_gradients)
        # Sum gradients from all nodes
        # for gradients in collected_gradients:
        #     for idx, grad in enumerate(gradients):
        #         aggregated_gradients[idx] += grad
        # Average the gradients
        # self.aggregated_gradients = [
        #     grad / num_nodes for grad in aggregated_gradients]
        self.aggregated_gradients = collected_gradients
        # Update global model parameters
        # self.global_model.train()
        # # with torch.no_grad():
        # for param, grad in zip(self.global_model.parameters(), aggregated_gradients):
        #     # param -= 0.001 * grad  # Update rule with learning rate 0.01
        #     param.grad = grad
        # self.optimizer.step()

    def communicate(self):
        """
        Sends the updated model parameters to all other clients.
        """
        for client_id in range(self.num_clients):
            # if client_id != self.client_id:
            # print("communicate, self.id, client id",
            #       self.client_id, client_id, self.joint_clients)
            if (client_id in self.joint_clients) and (client_id != self.client_id):
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
                    copy.deepcopy(self.global_model.state_dict()))
                print(
                    f"Client {self.client_id} sent model parameters to Client {client_id}."
                )

                # update the parameter if receive others'
                if self.received_models_q.qsize() != 0:
                    tmp_received_models = []
                    for _ in range(self.received_models_q.qsize()):
                        tmp_model = self.received_models_q.get()
                        tmp_received_models.append(tmp_model)
                        self.received_models.put(tmp_model)
                    # tmp_received_models.append(self.global_model)
                    # state_dicts = [model.state_dict()
                    #                for model in tmp_received_models]
                    state_dicts = tmp_received_models
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

                print(
                    "target client ",
                    target_client.client_id,
                    target_client.received_models.qsize(),
                    target_client.received_models_q.qsize(),
                    len(target_client.joint_clients),
                )

    def average_models(self):
        """
        Averages the received models to update the local global model.
        """
        while True:
            # print(
            #     "client ",
            #     self.client_id,
            #     self.received_models.qsize(),
            #     self.received_models_q.qsize(),
            #     len(self.joint_clients),
            # )
            # time.sleep(0.1)
            # if self.received_models_q.qsize() == len(self.joint_clients):
            #     break
            if (self.received_models.qsize() + self.received_models_q.qsize()) == len(
                self.joint_clients
            ):
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
            for gradients in self.aggregated_gradients:
                for param, grad in zip(
                    self.global_model.parameters(), gradients
                ):
                    # param -= 0.001 * grad  # Update rule with learning rate 0.01
                    # if param.grad is not None:
                    #     param.data -= grad
                    param.grad = grad
                self.optimizer.step()
            # self.received_models = Queue()
            # print(f"client {self.client_id}, averge model 1")
        else:
            for _ in range(self.received_models_q.qsize()):
                self.received_models.put(self.received_models_q.get())
            print(
                f"Client {self.client_id} received {self.received_models.qsize()} model parameters."
            )
            # # if not self.received_models:
            # if num_models == 0:
            #     return  # No models received
            # Include own model in averaging
            self.received_models.put(self.global_model.state_dict())
            # Average parameters using state_dict
            # Collect state_dicts from all models
            state_dicts = [
                self.received_models.get()
                for i in range(self.received_models.qsize())
            ]
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
            # Clear received models for the next round
            # self.received_models = Queue()
            self.global_model.train()
            for gradients in self.aggregated_gradients:
                for param, grad in zip(
                    self.global_model.parameters(), gradients
                ):
                    # if param.grad is not None:
                    #     param.data -= grad  # Modify the data directly
                    # param -= grad  # Update rule with learning rate 0.01
                    param.grad = grad
                self.optimizer.step()
        print(f"client {self.client_id}, averge model complete.")


# Define test function


def test_global_model(global_model, test_loader):
    global_model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            output = global_model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            # print(predicted)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    average_loss = total_loss / total
    return accuracy, average_loss


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    f = open("./results/res_{}_{}_{}.txt".format(num_clients, num_rounds,
             "-".join([str(i) for i in num_nodes_list])), "a+")
    # Initialize clients' global models (None at the start)
    clients_global_models = [VGG16() for _ in range(num_clients)]
    # clients_global_models = [ComplexCNN() for _ in range(num_clients)]
    joint_clients = customize_topology()
    # Lists to store accuracy and loss trends
    # Prepare test loader
    test_loader = DataLoader(
        test_dataset_full, batch_size=batch_size, shuffle=False)

    best_acc, best_round = 0.0, 0
    accuracy_list = []
    loss_list = []

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")
        # Create a list to hold clients for this round
        clients_list = []
        # Create clients and their nodes for this round
        for i, c_indices in enumerate(client_indices):
            # node_indices_list = create_noniid_splits(train_dataset, c_indices, 4)
            node_indices_list = create_iid_splits(
                train_dataset, c_indices, num_nodes_list[i]
            )

            # Use the previous global_model if exists
            global_model = clients_global_models[i]
            client = Client(
                i,
                node_indices_list,
                train_dataset,
                num_clients,
                clients_list,
                joint_clients[i],
                global_model=global_model,
            )
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
        clients_global_models.clear()
        clients_global_models = [
            client.global_model for client in clients_list]
        # Optionally, you can evaluate the global model here
        # For example, test accuracy on a validation set
        # Store the updated global models for the next round
        # clients_global_models = [client.global_model for client in clients_list]
        # Evaluate the global model (using the first client's model)
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
        del clients_list[:]
        torch.cuda.empty_cache()
        # for i in range(len(clients_list)):
        #     del clients_list[i]
