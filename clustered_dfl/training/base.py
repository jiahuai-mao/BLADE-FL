from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from clustered_dfl.types import ClientMetadata


ModelFactory = Callable[[], nn.Module]


@dataclass
class ClientState:
    client_id: int
    train_loader: DataLoader
    metadata: ClientMetadata
    model_vector: torch.Tensor | None = None
    local_step: int = 0


@dataclass
class TrainingConfig:
    algorithm: str
    rounds: int = 100
    events: int = 1000
    local_epochs: int = 1
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    eval_interval: int = 10
    device: str = "cpu"
    mixing_interval: int = 5
    seed: int = 0
    max_async_update_skew: int = 1


@dataclass
class RunArtifacts:
    metrics: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
