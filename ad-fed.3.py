# federated_learning_simulation.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import threading
import copy
import os
from queue import Queue
from torch.utils.data import DataLoader, Subset

# Set random seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_clients = 4
num_nodes_list = [6, 6, 6, 6]  # Number of nodes per client

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the FashionMNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

# Run the simulation for 200 rounds
num_rounds = 200
total_samples = len(train_dataset)
fraction = 0.3  # Use 30% of the dataset
num_samples = int(total_samples * fraction)
total_indices = list(range(total_samples))

# Shuffle and split indices among clients
random.shuffle(total_indices)
split_indices = np.array_split(total_indices, num_clients)
client_indices = [indices.tolist() for indices in split_indices]

# Prepare test dataset using a portion of the training data
test_dataset_full = Subset(
    train_dataset, total_indices[:int(total_samples * 0.3)]
)

# Define the topology among clients


def customize_topology():
    topology = []
    for i in range(num_clients):
        topology.append(
            [(i - 1 + num_clients) % num_clients,
             (i + 1 + num_clients) % num_clients]
        )
    return topology

# Create IID splits among nodes


def create_iid_splits(dataset, indices, num_nodes):
    node_indices = []
    random.shuffle(indices)
    split_indices = np.array_split(indices, num_nodes)
    for subset in split_indices:
        node_indices.append(subset.tolist())
    return node_indices

# Define the ComplexCNN model with improvements


class ComplexCNN(nn.Module):
    def __init__(self, input_channel=1, num_classes=10):
        super(ComplexCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Function to initialize weights


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

# Node class representing individual nodes


class Node(threading.Thread):
    def __init__(self, node_id, data_indices, dataset, global_model):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.data_loader = DataLoader(
            Subset(dataset, data_indices), batch_size=32, shuffle=True, num_workers=2
        )
        self.local_model = copy.deepcopy(global_model).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.num_local_epochs = 5  # Number of local training epochs
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.9)
        self.updated_model = None  # Placeholder for the updated model

    def run(self):
        # Local training
        self.local_model.train()
        for epoch in range(self.num_local_epochs):
            for data, target in self.data_loader:
                data = data.to(device)
                target = target.to(device)
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        # After training, save the updated model state_dict
        self.updated_model = copy.deepcopy(self.local_model.state_dict())
        print(f"{self.node_id} completed local training.")

# Client class representing individual clients


class Client(threading.Thread):
    def __init__(self, client_id, node_indices_list, dataset, num_clients, clients_list, joint_clients, global_model=None):
        threading.Thread.__init__(self)
        self.client_id = client_id
        self.nodes = []
        self.dataset = dataset
        self.global_model = global_model or ComplexCNN()
        self.global_model.apply(weights_init)
        self.global_model.to(device)
        self.num_clients = num_clients
        self.received_models_q = Queue()
        self.clients_list = clients_list
        self.joint_clients = joint_clients
        # Initialize nodes
        for i, node_indices in enumerate(node_indices_list):
            node = Node(
                f"Client{client_id}_Node{i}",
                node_indices,
                dataset,
                self.global_model
            )
            self.nodes.append(node)

    def run(self):
        # Start all nodes under this client
        for node in self.nodes:
            node.start()
        # Wait for all nodes to complete
        for node in self.nodes:
            node.join()
        print(
            f"Client {self.client_id} collected all updated models from nodes.")
        # Aggregate models and update the global model
        collected_models = [node.updated_model for node in self.nodes]
        self.aggregate_and_update(collected_models)
        print(f"Client {self.client_id} has updated the global model.")
        # Communicate updated model to other clients
        self.communicate()
        # Average received models from other clients
        self.average_models()
        print(
            f"Client {self.client_id} updated global model by averaging with other clients.")

    def aggregate_and_update(self, collected_models):
        # Average models from nodes
        state_dicts = collected_models
        param_keys = state_dicts[0].keys()
        averaged_state_dict = {}
        num_models = len(state_dicts)
        for key in param_keys:
            params = [state_dict[key] for state_dict in state_dicts]
            stacked_params = torch.stack(params, dim=0)
            if torch.is_floating_point(stacked_params):
                # Floating point tensors can be averaged
                averaged_param = torch.mean(stacked_params, dim=0)
            else:
                # For integer tensors, take the value from the first model
                averaged_param = stacked_params[0]
            averaged_state_dict[key] = averaged_param
        self.global_model.load_state_dict(averaged_state_dict)

    def communicate(self):
        # Send updated global model to neighboring clients
        for client_id in self.joint_clients:
            if client_id != self.client_id:
                target_client = self.clients_list[client_id]
                target_client.received_models_q.put(
                    copy.deepcopy(self.global_model.state_dict()))
                print(
                    f"Client {self.client_id} sent model to Client {client_id}.")

    def average_models(self):
        # Average received global models from other clients
        num_models_to_receive = len(self.joint_clients) - 1  # Exclude self
        received_state_dicts = []
        while len(received_state_dicts) < num_models_to_receive:
            if not self.received_models_q.empty():
                model_state_dict = self.received_models_q.get()
                received_state_dicts.append(model_state_dict)
        # Include own global model in averaging
        received_state_dicts.append(self.global_model.state_dict())
        # Average parameters
        param_keys = received_state_dicts[0].keys()
        averaged_state_dict = {}
        num_models = len(received_state_dicts)
        for key in param_keys:
            params = [state_dict[key] for state_dict in received_state_dicts]
            stacked_params = torch.stack(params, dim=0)
            if torch.is_floating_point(stacked_params):
                # Floating point tensors can be averaged
                averaged_param = torch.mean(stacked_params, dim=0)
            else:
                # For integer tensors, take the value from the first model
                averaged_param = stacked_params[0]
            averaged_state_dict[key] = averaged_param
        # Update global model
        self.global_model.load_state_dict(averaged_state_dict)


# Define test function


def test_global_model(global_model, test_loader):
    global_model.eval()
    global_model.to(device)
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = global_model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    average_loss = total_loss / total
    return accuracy, average_loss


# Initialize clients' global models
clients_global_models = [ComplexCNN().to(device) for _ in range(num_clients)]
for model in clients_global_models:
    model.apply(weights_init)

joint_clients = customize_topology()

# Prepare test loader
test_loader = DataLoader(test_dataset_full, batch_size=32,
                         shuffle=False, num_workers=2)

if not os.path.exists('./results'):
    os.makedirs('./results')

best_acc, best_round = 0.0, 0
accuracy_list = []
loss_list = []

for round_num in range(num_rounds):
    print(f"\n=== Round {round_num + 1} ===")
    # Create a list to hold clients for this round
    clients_list = []
    # Create clients and their nodes for this round
    for i, c_indices in enumerate(client_indices):
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
            global_model=global_model
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
    clients_global_models = [client.global_model for client in clients_list]
    # Evaluate the global model (using the first client's model)
    global_model = clients_global_models[0]
    accuracy, avg_loss = test_global_model(global_model, test_loader)
    accuracy_list.append(accuracy)
    loss_list.append(avg_loss)
    if accuracy >= best_acc:
        best_acc = accuracy
        best_round = round_num + 1
    print(
        f"Round {round_num + 1}: Test Accuracy: {accuracy*100:.2f}%, Test Loss: {avg_loss:.4f}"
    )

# Write results to file
filename = "./results/res_{}.txt".format(seed)
with open(filename, "wt") as f:
    for i in range(len(accuracy_list)):
        acc = accuracy_list[i]
        loss = loss_list[i]
        f.write("round {}, acc {:.4f}, loss {:.4f}, best acc {:.4f}, best round {}\n".format(
            i + 1, acc, loss, best_acc, best_round))
