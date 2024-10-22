import threading
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import copy
import time
from queue import Queue

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    transforms.Normalize((0.5,), (0.5,))
])


# load the fmnist training dataset
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
test_dataset_full = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Total number of samples in the dataset
# num_samples = len(train_dataset)
# indices = list(range(num_samples))
# random.shuffle(indices)

# Split the dataset indices into 4 equal parts for the 4 clients
# client_indices = np.array_split(indices, 24)
total_samples = len(train_dataset)
fraction = 0.25  # Change to 0.3 for 30%
num_samples = int(total_samples * fraction)

# Generate random indices for the subset
indices = list(range(total_samples))
# random.shuffle(indices)
# subset_indices = indices[:num_samples]


client_indices = []
for _ in range(24):
    random.shuffle(indices)
    subset_indices = indices[:num_samples]
    client_indices.append(subset_indices)

parameters_lock = [threading.Lock() for _ in range(4)]
numpara_lock = [threading.Lock() for _ in range(4)]


def customize_topology():
    topology = list()
    for i in range(24):
        topology.append([(i-1+24) % 24, (i+1+24) % 24])
    return topology


def create_iid_splits(dataset, indices, num_nodes):
    node_indices = [list() for _ in range(num_nodes)]
    for id in indices:
        # x = np.random.randint(0, 4)
        node_indices[0].append(id)
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


# Example usage
n = 5  # Size of the matrix
W_k = generate_doubly_stochastic_matrix(n)

print("Doubly stochastic matrix A:")
# print(A)
print("\nRow sums:", W_k.sum(axis=1))
print("Column sums:", W_k.sum(axis=0))


class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for CIFAR-10.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ComplexCNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(ComplexCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.Tanh(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# class Node(threading.Thread):
#     """
#     Represents a node in the network.
#     Each node performs local training and computes gradients.
#     """

#     def __init__(self, node_id, data_indices, dataset, global_model, aggregator_queue):
#         threading.Thread.__init__(self)
#         self.node_id = node_id
#         self.data_loader = DataLoader(
#             Subset(dataset, data_indices),
#             batch_size=32,
#             shuffle=True
#         )
#         self.local_model = copy.deepcopy(global_model).cuda()
#         self.criterion = nn.CrossEntropyLoss()
#         self.gradients = None  # Placeholder for storing gradients
#         # Queue to send gradients to aggregator
#         self.aggregator_queue = aggregator_queue

#     def run(self):
#         print(f"{self.node_id} starting local training.")
#         # Perform forward and backward passes to compute gradients
#         # self.local_model = self.local_model.cuda()
#         self.local_model.train()
#         self.local_model.zero_grad()
#         for data, target in self.data_loader:
#             data = data.cuda()
#             target = target.cuda()
#             output = self.local_model(data)
#             loss = self.criterion(output, target)
#             loss.backward()
#             # break  # For demonstration, we only process one batch
#         # Extract gradients
#         self.gradients = [param.grad.clone()
#                           for param in self.local_model.cpu().parameters()]
#         # Send gradients to aggregator
#         self.aggregator_queue.put(self.gradients)
#         print(f"{self.node_id} sent gradients to aggregator.")


class Client(threading.Thread):
    """
    Represents a client that manages its nodes and communicates with other clients.
    """

    def __init__(self, client_id, node_indices_list, dataset, num_clients, clients_list, joint_clients, global_model=None):
        threading.Thread.__init__(self)
        self.client_id = client_id
        # self.nodes = []
        self.gradients = None
        self.dataset = dataset
        # Initialize global model if not provided
        # self.global_model = global_model if global_model else SimpleCNN()
        self.global_model = global_model if global_model else ComplexCNN(
            input_channel=1)
        self.global_model = self.global_model.cuda()
        self.num_clients = num_clients
        # self.received_models = []  # List to store models received from other clients
        self.received_models = Queue()
        self.received_models_q = Queue()
        self.aggregator_queue = Queue()  # Queue to collect gradients from nodes
        self.clients_list = clients_list  # Reference to other clients
        self.joint_clients = joint_clients
        # self.optimizer = torch.optim.Adam(
        #     self.global_model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.global_model.parameters(
        ), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.data_loader = DataLoader(
            Subset(dataset, node_indices_list[0]),
            batch_size=32,
            shuffle=True
        )

    def run(self):
        print(f"Client {self.client_id} starting.")
        # Start all nodes under this client
        # for node in self.nodes:
        #     node.start()
        # Collect gradients from nodes
        # collected_gradients = []
        # for _ in self.nodes:
        #     gradients = self.aggregator_queue.get()
        #     collected_gradients.append(gradients)
        # # Wait for all nodes to complete
        # for node in self.nodes:
        #     node.join()
        self.global_model.train()
        self.global_model.zero_grad()
        for data, target in self.data_loader:
            data = data.cuda()
            target = target.cuda()
            output = self.global_model(data)
            loss = self.criterion(output, target)
            # self.optimizer.zero_grad()
            loss.backward()
            # self.optimizer.step()
        # Extract gradients
        self.gradients = [param.grad.clone()
                          for param in self.global_model.cpu().parameters()]
        # # Send gradients to aggregator
        # self.aggregator_queue.put(self.gradients)
        # print(f"Client {self.client_id} collected all gradients.")
        # Aggregate gradients and update the global model
        # self.aggregate_and_update(collected_gradients)
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
        # Initialize aggregated gradients
        aggregated_gradients = [torch.zeros_like(
            param) for param in self.global_model.parameters()]
        num_nodes = len(collected_gradients)
        # Sum gradients from all nodes
        for gradients in collected_gradients:
            for idx, grad in enumerate(gradients):
                aggregated_gradients[idx] += grad
        # Average the gradients
        aggregated_gradients = [
            grad / num_nodes for grad in aggregated_gradients]
        # Update global model parameters
        self.global_model.train()
        # with torch.no_grad():
        for param, grad in zip(self.global_model.parameters(), aggregated_gradients):
            # param -= 0.001 * grad  # Update rule with learning rate 0.01
            param.grad = grad
        self.optimizer.step()

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
num_clients = 24

# Initialize clients' global models (None at the start)
clients_global_models = [None] * num_clients
joint_clients = customize_topology()
# Lists to store accuracy and loss trends
accuracy_list = []
loss_list = []
# Prepare test loader
test_loader = DataLoader(test_dataset_full, batch_size=32, shuffle=False)
# Run the simulation for 10 rounds
num_rounds = 100


for round_num in range(num_rounds):
    print(f"\n=== Round {round_num + 1} ===")
    # Create a list to hold clients for this round
    clients_list = []
    # Create clients and their nodes for this round
    for i, c_indices in enumerate(client_indices):
        # node_indices_list = create_noniid_splits(train_dataset, c_indices, 4)
        node_indices_list = create_iid_splits(train_dataset, c_indices, 1)
        # Use the previous global_model if exists
        global_model = clients_global_models[i]
        client = Client(i, node_indices_list, train_dataset,
                        num_clients, clients_list, joint_clients[i], global_model=global_model)
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
    clients_global_models = [client.global_model for client in clients_list]
    # Evaluate the global model (using the first client's model)
    global_model = clients_global_models[0]
    accuracy, avg_loss = test_global_model(global_model, test_loader)
    accuracy_list.append(accuracy)
    loss_list.append(avg_loss)
    print(
        f"Round {round_num + 1}: Test Accuracy: {accuracy*100:.2f}%, Test Loss: {avg_loss:.4f}")
