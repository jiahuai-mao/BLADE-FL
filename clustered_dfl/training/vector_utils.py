from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def model_to_vector(model: nn.Module) -> torch.Tensor:
    return parameters_to_vector(model.parameters()).detach().cpu().clone()


def vector_to_model(vector: torch.Tensor, model: nn.Module) -> nn.Module:
    target = vector.detach().to(next(model.parameters()).device)
    vector_to_parameters(target, model.parameters())
    return model


def average_vectors(vectors: Sequence[torch.Tensor], weights: Sequence[float] | None = None) -> torch.Tensor:
    if not vectors:
        raise ValueError("vectors must not be empty.")
    stacked = torch.stack([vector.detach().cpu() for vector in vectors])
    if weights is None:
        return stacked.mean(dim=0)
    weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    total = weight_tensor.sum()
    if float(total) <= 0.0:
        raise ValueError("sum(weights) must be positive.")
    weight_tensor = weight_tensor / total
    return torch.sum(stacked * weight_tensor.reshape(-1, 1), dim=0)


def initial_model_vector(model_factory, seed: int, device: str = "cpu") -> torch.Tensor:
    torch.manual_seed(seed)
    model = model_factory().to(device)
    return model_to_vector(model)
