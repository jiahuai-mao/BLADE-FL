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
    clipprox_mu: float = 0.0
    clipprox_clip_norm: float = 1.0
    cluster_delta_clip_norm: float = 0.0
    lr_scheduler: str = "none"
    sync_comm_bandwidth_proxy: float = 1_000_000.0


@dataclass
class RunArtifacts:
    metrics: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    extra_tables: dict[str, list[dict]] = field(default_factory=dict)


def log_progress(row: dict, total_steps: int) -> None:
    step = int(row.get("step", 0) or 0)
    if step <= 0 or total_steps <= 0:
        return

    width = 30
    ratio = min(max(step / total_steps, 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)

    accuracy = _format_metric(row.get("test_accuracy"))
    macro_f1 = _format_metric(row.get("test_macro_f1"))
    best_accuracy = _format_metric(row.get("best_accuracy"))
    train_loss = _format_metric(row.get("train_loss"))
    virtual_time = _format_metric(row.get("virtual_time"))
    print(
        "[progress] "
        f"algorithm={row.get('algorithm')} "
        f"[{bar}] {step}/{total_steps} ({ratio * 100:.1f}%) "
        f"acc={accuracy} macro_f1={macro_f1} best={best_accuracy} "
        f"loss={train_loss} virtual_time={virtual_time}",
        flush=True,
    )


def _format_metric(value) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.4f}"
