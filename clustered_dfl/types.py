from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClientMetadata:
    """Compact descriptor used only for topology construction."""

    client_id: int
    num_samples: int
    label_counts: list[int]
    train_time: float


@dataclass(frozen=True)
class Topology:
    clusters: list[list[int]]
    leaders: list[int]
    leader_edges: list[tuple[int, int]]
    client_to_cluster: dict[int, int]
