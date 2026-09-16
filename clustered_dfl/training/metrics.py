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
    model: nn.Module | None = None,
) -> tuple[float | None, float | None, float | None]:
    if test_loader is None:
        return None, None, None
    model = model if model is not None else model_factory().to(device)
    model.to(device)
    vector_to_model(model_vector, model)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    non_blocking = str(device).startswith("cuda")
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = None
    for features, labels in test_loader:
        features = features.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        logits = model(features)
        total_loss += float(criterion(logits, labels).item())
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
        confusion = _update_confusion(confusion, predictions.detach(), labels.detach(), logits.shape[1])
    if total == 0:
        return None, None, None
    macro_f1 = _macro_f1_from_confusion(confusion)
    return total_loss / total, correct / total, macro_f1


def _update_confusion(confusion: torch.Tensor | None, predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if confusion is None:
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=predictions.device)
    encoded = labels.to(torch.long) * num_classes + predictions.to(torch.long)
    counts = torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return confusion + counts.to(dtype=confusion.dtype)


def _macro_f1_from_confusion(confusion: torch.Tensor | None) -> float:
    if confusion is None:
        return 0.0
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    support = confusion.sum(dim=1)
    denom = (2.0 * tp) + fp + fn
    scores = torch.where(denom > 0, (2.0 * tp) / denom, torch.zeros_like(denom))
    active = support > 0
    if not bool(active.any()):
        return 0.0
    return float(scores[active].mean().item())


def weighted_global_model(vectors: Sequence[torch.Tensor], sample_counts: Sequence[int]) -> torch.Tensor:
    return average_vectors(vectors, [float(count) for count in sample_counts])


def model_divergence(vectors: Sequence[torch.Tensor], global_vector: torch.Tensor) -> float:
    if not vectors:
        return 0.0
    device = global_vector.device
    distances = [float(torch.norm(vector.detach().to(device) - global_vector.detach()).item()) for vector in vectors]
    return sum(distances) / len(distances)
