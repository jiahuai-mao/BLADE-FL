from __future__ import annotations

import torch
from torch import nn

from .base import ClientState, TrainingConfig
from .vector_utils import model_to_vector, vector_to_model


def train_client_from_vector(
    model_factory,
    start_vector: torch.Tensor,
    client: ClientState,
    config: TrainingConfig,
    prox_reference_vector: torch.Tensor | None = None,
    prox_mu: float = 0.0,
    model: nn.Module | None = None,
) -> tuple[torch.Tensor, float]:
    device = config.device
    model = model if model is not None else model_factory().to(device)
    model.to(device)
    vector_to_model(start_vector, model)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, client, config)
    criterion = nn.CrossEntropyLoss()
    prox_reference_params = _reference_parameter_tensors(prox_reference_vector, model) if prox_reference_vector is not None and prox_mu > 0.0 else None
    non_blocking = str(device).startswith("cuda")
    total_loss = 0.0
    total_seen = 0
    for _ in range(config.local_epochs):
        for features, labels in client.train_loader:
            features = features.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            if prox_reference_params is not None:
                prox_loss = sum(
                    torch.sum((param - reference) ** 2)
                    for param, reference in zip(model.parameters(), prox_reference_params)
                )
                loss = loss + 0.5 * prox_mu * prox_loss
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size
    mean_loss = total_loss / total_seen if total_seen else 0.0
    return model_to_vector(model), mean_loss


def _make_scheduler(optimizer: torch.optim.Optimizer, client: ClientState, config: TrainingConfig):
    scheduler_name = config.lr_scheduler.lower()
    if scheduler_name in {"", "none", "constant"}:
        return None
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported lr_scheduler: {config.lr_scheduler}")
    steps_per_epoch = len(client.train_loader)
    total_steps = max(1, int(config.local_epochs) * max(1, steps_per_epoch))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


def _reference_parameter_tensors(vector: torch.Tensor, model: nn.Module) -> list[torch.Tensor]:
    device = next(model.parameters()).device
    vector = vector.detach().to(device)
    tensors = []
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        tensors.append(vector[offset : offset + numel].view_as(param))
        offset += numel
    return tensors
