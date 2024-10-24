# import threading
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import random
# import copy
# import time

# # Define transformations for the dataset
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # Load the FashionMNIST training dataset
# train_dataset = torchvision.datasets.FashionMNIST(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transform
# )

# # Define a simple model for demonstration purposes


# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc = nn.Linear(64 * 64, 10)

#     def forward(self, x):
#         x = x.view(-1, 64 * 64)
#         return self.fc(x)

# # D-PSGD Algorithm Implementation


# class DPSGDClient(threading.Thread):
#     def __init__(self, client_id, model, dataset, num_epochs=1, lr=0.01):
#         threading.Thread.__init__(self)
#         self.client_id = client_id
#         self.model = copy.deepcopy(model)
#         self.dataset = dataset
#         self.num_epochs = num_epochs
#         self.lr = lr
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
#         self.criterion = nn.CrossEntropyLoss()
#         self.neighbors = []

#     def run(self):
#         for epoch in range(self.num_epochs):
#             self.local_update()
#             self.communicate_with_neighbors()

#     def local_update(self):
#         self.model.train()
#         data_loader = torch.utils.data.DataLoader(
#             self.dataset, batch_size=32, shuffle=True)
#         for data, target in data_loader:
#             self.optimizer.zero_grad()
#             output = self.model(data)
#             loss = self.criterion(output, target)
#             loss.backward()
#             self.optimizer.step()

#     def communicate_with_neighbors(self):
#         # Average parameters with neighbors
#         for param in self.model.parameters():
#             param_sum = param.data.clone()
#             for neighbor in self.neighbors:
#                 param_sum += neighbor.model.state_dict()[param.name]
#             param.data = param_sum / (len(self.neighbors) + 1)

# # AD-PSGD Algorithm Implementation


# class ADPSGDClient(threading.Thread):
#     def __init__(self, client_id, model, dataset, num_epochs=1, lr=0.01):
#         threading.Thread.__init__(self)
#         self.client_id = client_id
#         self.model = copy.deepcopy(model)
#         self.dataset = dataset
#         self.num_epochs = num_epochs
#         self.lr = lr
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
#         self.criterion = nn.CrossEntropyLoss()
#         self.neighbors = []

#     def run(self):
#         for epoch in range(self.num_epochs):
#             self.local_update()
#             self.asynchronous_communication()

#     def local_update(self):
#         self.model.train()
#         data_loader = torch.utils.data.DataLoader(
#             self.dataset, batch_size=32, shuffle=True)
#         for data, target in data_loader:
#             self.optimizer.zero_grad()
#             output = self.model(data)
#             loss = self.criterion(output, target)
#             loss.backward()
#             self.optimizer.step()

#     def asynchronous_communication(self):
#         for param in self.model.parameters():
#             param_sum = param.data.clone()
#             for neighbor in self.neighbors:
#                 neighbor_param = neighbor.model.state_dict()[param.name]
#                 param_sum += neighbor_param
#                 # Simulate asynchronous behavior
#                 time.sleep(random.uniform(0.01, 0.1))
#             param.data = param_sum / (len(self.neighbors) + 1)

# # SGP Algorithm Implementation


# class SGPClient(threading.Thread):
#     def __init__(self, client_id, model, dataset, weight_matrix, num_epochs=1, lr=0.01):
#         threading.Thread.__init__(self)
#         self.client_id = client_id
#         self.model = copy.deepcopy(model)
#         self.dataset = dataset
#         self.weight_matrix = weight_matrix
#         self.num_epochs = num_epochs
#         self.lr = lr
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
#         self.criterion = nn.CrossEntropyLoss()
#         self.neighbors = []

#     def run(self):
#         for epoch in range(self.num_epochs):
#             self.local_update()
#             self.push_gradients()

#     def local_update(self):
#         self.model.train()
#         data_loader = torch.utils.data.DataLoader(
#             self.dataset, batch_size=32, shuffle=True)
#         for data, target in data_loader:
#             self.optimizer.zero_grad()
#             output = self.model(data)
#             loss = self.criterion(output, target)
#             loss.backward()
#             self.optimizer.step()

#     def push_gradients(self):
#         for param in self.model.parameters():
#             weighted_sum = param.data * \
#                 self.weight_matrix[self.client_id, self.client_id]
#             for neighbor_id in self.neighbors:
#                 neighbor_weight = self.weight_matrix[self.client_id, neighbor_id]
#                 neighbor_param = neighbor_id.model.state_dict()[param.name]
#                 weighted_sum += neighbor_weight * neighbor_param
#             param.data = weighted_sum


# # Example Usage
# model = SimpleModel()
# clients = [DPSGDClient(i, model, train_dataset) for i in range(4)]

# # Assign neighbors for demonstration purposes
# for i in range(len(clients)):
#     clients[i].neighbors = [
#         clients[(i - 1) % len(clients)], clients[(i + 1) % len(clients)]]

# # Start all clients
# for client in clients:
#     client.start()

# # Wait for all clients to complete
# for client in clients:
#     client.join()

import numpy as np


def generate_doubly_stochastic_matrix(n, tol=1e-10, max_iter=1000):
    """
    Generates an arbitrary n x n doubly stochastic matrix using the Sinkhorn-Knopp algorithm.

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
n = 5
W_k = generate_doubly_stochastic_matrix(n)
print("Doubly stochastic matrix A:")
print(W_k)
print("\nRow sums:", W_k.sum(axis=1))
print("Column sums:", W_k.sum(axis=0))
