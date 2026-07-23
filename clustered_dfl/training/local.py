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
) -> tuple[torch.Tensor, float]:
    device = config.device
    model = model_factory().to(device)
    vector_to_model(start_vector, model)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_seen = 0
    for _ in range(config.local_epochs):
        for features, labels in client.train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size
    mean_loss = total_loss / total_seen if total_seen else 0.0
    return model_to_vector(model), mean_loss
