# dpsgd.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import threading
import time
from torch.utils.data import DataLoader, Subset

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(
    './data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    './data', train=False, transform=transform)

# Number of nodes
num_nodes = 4

# Define neighbors for each node in a ring topology
topology = {i: [(i - 1) % num_nodes, (i + 1) % num_nodes] for i in range(num_nodes)}

# Split the dataset indices among nodes
indices = np.arange(len(train_dataset))
np.random.shuffle(indices)
split_indices = np.array_split(indices, num_nodes)

# Create DataLoaders for each node
node_data_loaders = []
batch_size = 64
for i in range(num_nodes):
    subset = Subset(train_dataset, split_indices[i])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    node_data_loaders.append(loader)

# Create a DataLoader for the test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Simple Feedforward Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model on local data
def train(model, data_loader, optimizer, criterion, epochs=1):
    model.train()
    for _ in range(epochs):
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Function to evaluate the model
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    accuracy = 100.0 * correct / total
    return accuracy

class Node(threading.Thread):
    def __init__(self, node_id, model, data_loader, neighbors):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.model = model
        self.data_loader = data_loader
        self.neighbors = neighbors
        self.received_models = []
        self.lock = threading.Lock()
        self.stop_signal = False

    def run(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        epochs = 1  # Local epochs

        while not self.stop_signal:
            # Local Training
            train(self.model, self.data_loader, optimizer, criterion, epochs)

            # Communication Step (Synchronous)
            self.communicate()

            # Average Models
            self.average_models()

    def communicate(self):
        # Send model to neighbors
        for neighbor in self.neighbors:
            neighbor_node = nodes[neighbor]
            with neighbor_node.lock:
                neighbor_node.received_models.append(self.get_model_copy())

    def average_models(self):
        with self.lock:
            if self.received_models:
                # Include own model
                self.received_models.append(self.get_model_copy())
                # Average parameters
                new_state_dict = {}
                for key in self.model.state_dict().keys():
                    params = [state_dict[key] for state_dict in self.received_models]
                    new_state_dict[key] = sum(params) / len(params)
                self.model.load_state_dict(new_state_dict)
                self.received_models = []

    def get_model_copy(self):
        return {key: value.clone() for key, value in self.model.state_dict().items()}

# Initialize nodes
nodes = []
for i in range(num_nodes):
    model = Net()
    node = Node(
        node_id=i,
        model=model,
        data_loader=node_data_loaders[i],
        neighbors=topology[i]
    )
    nodes.append(node)

# Start nodes
for node in nodes:
    node.start()

# Run for a certain duration
run_duration = 60  # seconds
print(f"Running D-PSGD for {run_duration} seconds...")
time.sleep(run_duration)

# Signal nodes to stop
for node in nodes:
    node.stop_signal = True

# Wait for all nodes to finish
for node in nodes:
    node.join()

# Evaluate each node's model
for i, node in enumerate(nodes):
    accuracy = evaluate(node.model, test_loader)
    print(f'Node {i} Model Accuracy: {accuracy:.2f}%')
