from __future__ import annotations

from functools import reduce
from operator import mul

import torch
from torch import nn


DATASET_SPECS = {
    "synthetic": ((20,), 10),
    "mnist": ((1, 28, 28), 10),
    "fashionmnist": ((1, 28, 28), 10),
    "cifar10": ((3, 32, 32), 10),
    "cifar100": ((3, 32, 32), 100),
    "svhn": ((3, 32, 32), 10),
    "hhar": ((12,), 6),
}


class FlattenMLP(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        input_dim = int(reduce(mul, input_shape, 1))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CIFARCNNBN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 64),
            self._conv_block(64, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            self._conv_block(128, 128),
            nn.MaxPool2d(2),
            self._conv_block(128, 256),
            self._conv_block(256, 256),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class HHAR1DCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"HHAR1DCNN expects input with shape (batch, channels, time), got {tuple(x.shape)}.")
        return self.classifier(self.features(x))


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, track_running_stats=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class CIFARResNet(nn.Module):
    def __init__(self, block: type[CIFARBasicBlock], layers: list[int], num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block: type[CIFARBasicBlock], out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet20(num_classes: int = 10) -> CIFARResNet:
    return CIFARResNet(CIFARBasicBlock, [3, 3, 3], num_classes=num_classes)


def create_model(dataset: str, hidden_dim: int = 128, model_name: str = "auto", hhar_input_channels: int = 3) -> nn.Module:
    dataset = dataset.lower()
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset for model creation: {dataset}")
    input_shape, num_classes = DATASET_SPECS[dataset]
    model_name = model_name.lower()
    if model_name == "auto":
        model_name = "cnnbn" if dataset in {"cifar10", "cifar100", "svhn"} else "mlp"
    if model_name in {"cnnbn", "cnn_bn", "cifarcnnbn"}:
        if len(input_shape) != 3 or input_shape[0] != 3:
            raise ValueError(f"CnnBN requires 3-channel image input, got dataset={dataset}.")
        return CIFARCNNBN(num_classes=num_classes)
    if model_name in {"resnet20", "cifar_resnet20"}:
        if len(input_shape) != 3 or input_shape[0] != 3:
            raise ValueError(f"ResNet20 requires 3-channel image input, got dataset={dataset}.")
        return resnet20(num_classes=num_classes)
    if model_name in {"hhar1dcnn", "hhar_1dcnn", "hhar_cnn1d", "cnn1d"}:
        if dataset != "hhar":
            raise ValueError(f"HHAR1DCNN requires dataset=hhar, got dataset={dataset}.")
        return HHAR1DCNN(in_channels=hhar_input_channels, num_classes=num_classes)
    if model_name not in {"mlp", "flatten_mlp", "flattenmlp"}:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return FlattenMLP(input_shape=input_shape, num_classes=num_classes, hidden_dim=hidden_dim)
