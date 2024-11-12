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

num_clients = 1
num_nodes_list = [1]

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
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

num_rounds = 2000
batch_size = 256
accumulation_steps = 1
total_samples = len(train_dataset)
fraction = 0.8
learning_rate = 0.01
lr_decay = 0.95
lr_decay_step = 50
num_samples = int(total_samples * fraction)
total_indices = list(range(total_samples))

random.shuffle(total_indices)

data_loader = DataLoader(
            Subset(train_dataset, total_indices[:num_samples]), batch_size=batch_size, shuffle=True
        )

# random.shuffle(total_indices)

# test_dataset_full = Subset(train_dataset, total_indices[:10000])
test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)



class ComplexCNN(nn.Module):
    def __init__(self, in_channels=1, hidden_size=200, num_classes=10):
        super(ComplexCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_size
        self.num_classes = num_classes

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=(
                5, 5), padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels *
                            2, kernel_size=(5, 5), padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=(self.hidden_channels * 2)
                            * (7 * 7), out_features=512, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features=512,
                            out_features=self.num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

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
    model = ComplexCNN()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_loss = 0
    test_loss = 0
    correct = 0
    
    for i in range(num_rounds):
        print("-------------round: ", i, "-------------")
        model.train()
        for data, target in data_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            _, predicted = output.max(1)
            
            correct = predicted.eq(target).sum().item()
            print("--train--acc: ", correct / batch_size, ", loss: ", loss.item()/batch_size)
            break
        
        accuracy, avg_loss = test_global_model(model, test_loader)
        print(
            f"Round {i + 1}: Test Accuracy: {accuracy*100:.2f}%, Test Loss: {avg_loss:.4f}, Learning rate: {learning_rate:.4f}"
        )
        if (i+1) % lr_decay_step == 0:
            learning_rate = learning_rate * lr_decay
        print()
            