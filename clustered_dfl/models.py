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


def create_model(dataset: str, hidden_dim: int = 128) -> nn.Module:
    dataset = dataset.lower()
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset for model creation: {dataset}")
    input_shape, num_classes = DATASET_SPECS[dataset]
    return FlattenMLP(input_shape=input_shape, num_classes=num_classes, hidden_dim=hidden_dim)
