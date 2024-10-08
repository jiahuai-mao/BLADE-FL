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
from tqdm import tqdm
from demo.demo.demoloader.vgg_bn import vgg19_bn

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

# Run the simulation for 10 rounds
num_rounds = 1000
batch_size = 512


def train(net, trainloader, optimizer, criterion):
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()

    print('Train Acc: %.3f%% (%d/%d) | Loss: %.3f' %
          (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

    return


def test(net, testloader, optimizer, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader):

            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

        print('Test Acc: %.3f%% (%d/%d)' %
              (100.*correct/total, correct, total))

    return


if __name__ == "__main__":
    # model = VGG16()
    model = vgg19_bn(input_channel=1, num_classes=10)
    print(model)
    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda()

    for epoch in range(num_rounds):
        train(model, data_loader, optimizer, criterion)
        test(model, test_loader, optimizer, criterion)
