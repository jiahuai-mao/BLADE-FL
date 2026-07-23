from __future__ import annotations

from clustered_dfl.training.base import ClientState
from clustered_dfl.types import Topology


def client_train_time(client: ClientState, local_epochs: int | float) -> float:
    return float(local_epochs) * float(client.metadata.num_samples)


def cluster_update_times(clients_by_id: dict[int, ClientState], topology: Topology, local_epochs: int | float) -> list[float]:
    times = []
    for cluster in topology.clusters:
        if not cluster:
            times.append(0.0)
            continue
        times.append(max(client_train_time(clients_by_id[cid], local_epochs) for cid in cluster))
    return times
