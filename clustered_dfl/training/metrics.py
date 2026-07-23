from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from .vector_utils import average_vectors, vector_to_model


@torch.no_grad()
def evaluate_vector(
    model_factory,
    model_vector: torch.Tensor,
    test_loader: DataLoader | None,
    device: str,
) -> tuple[float | None, float | None]:
    if test_loader is None:
        return None, None
    model = model_factory().to(device)
    vector_to_model(model_vector, model)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        total_loss += float(criterion(logits, labels).item())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    if total == 0:
        return None, None
    return total_loss / total, correct / total


def weighted_global_model(vectors: Sequence[torch.Tensor], sample_counts: Sequence[int]) -> torch.Tensor:
    return average_vectors(vectors, [float(count) for count in sample_counts])


def model_divergence(vectors: Sequence[torch.Tensor], global_vector: torch.Tensor) -> float:
    if not vectors:
        return 0.0
    distances = [float(torch.norm(vector.detach().cpu() - global_vector.detach().cpu()).item()) for vector in vectors]
    return sum(distances) / len(distances)
