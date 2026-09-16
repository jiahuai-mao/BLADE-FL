from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from clustered_dfl.clustering import (
    candidate_k_values,
    client_histogram,
    communication_proxy,
    compute_capacities,
    distribution_cost,
    global_histogram,
    waiting_cost,
)
from clustered_dfl.topology import Graph, select_connected_leaders
from clustered_dfl.types import ClientMetadata, Topology

from .clustered import ClusteredAsyncTrainingRunner


def build_mdfeel_topology(
    clients: list[ClientMetadata],
    k: int,
    feasible_graph: Graph,
    *,
    seed: int = 0,
    max_iter: int = 8,
) -> tuple[Topology, dict[str, Any]]:
    """Build an adapted MD-FEEL topology from data-distribution similarity.

    The baseline keeps the existing simulator's capacity and leader-connectivity
    constraints, but uses only client label-histogram similarity for assignment.
    This avoids reusing BLADE-FL's waiting/distribution balanced objective.
    """

    if not clients:
        raise ValueError("clients must not be empty.")
    if k <= 0 or k > len(clients):
        raise ValueError(f"k must be in [1, {len(clients)}], got {k}.")

    clients_by_id = {client.client_id: client for client in clients}
    capacities = compute_capacities(len(clients), k)
    global_hist = global_histogram(clients)
    histograms = {client.client_id: client_histogram(client) for client in clients}
    centroids = _initial_centroids(clients, histograms, global_hist, k)

    clusters: list[list[int]] = []
    for _ in range(max_iter):
        next_clusters = _assign_by_similarity(clients, histograms, centroids, capacities)
        next_centroids = _cluster_centroids(next_clusters, clients_by_id, histograms)
        if next_clusters == clusters:
            break
        clusters = next_clusters
        centroids = next_centroids

    if not clusters:
        clusters = _assign_by_similarity(clients, histograms, centroids, capacities)

    leaders, leader_edges, leader_score, connected = select_connected_leaders(clusters, clients_by_id, feasible_graph)
    if not connected:
        raise RuntimeError("adapted MD-FEEL topology failed to form a connected leader overlay.")

    topology = Topology(
        clusters=clusters,
        leaders=leaders,
        leader_edges=leader_edges,
        client_to_cluster={cid: cluster_idx for cluster_idx, cluster in enumerate(clusters) for cid in cluster},
    )
    metrics = _mdfeel_metrics(topology, clients_by_id, histograms, centroids, global_hist)
    metrics.update(
        {
            "K": k,
            "capacities": capacities,
            "leader_score": float(leader_score),
            "leader_edges": leader_edges,
            "connected": connected,
            "method": "adapted_mdfeel_similarity",
            "max_iter": max_iter,
            "seed": seed,
        }
    )
    return topology, metrics


def select_best_mdfeel_topology(
    clients: list[ClientMetadata],
    *,
    k_min: int,
    k_max: int | None,
    min_clients_per_cluster: int,
    lambda_communication: float,
    mixing_interval: int,
    feasible_graph: Graph,
    seed: int = 0,
    max_iter: int = 8,
) -> tuple[Topology, dict[str, Any]]:
    candidates = candidate_k_values(len(clients), k_min, min_clients_per_cluster, k_max)
    if not candidates:
        raise ValueError("no feasible K values for adapted MD-FEEL topology.")

    best: tuple[float, int, Topology, dict[str, Any]] | None = None
    k_search = []
    max_comm = max(
        communication_proxy(len(clients), k, [], 1.0, max(1, mixing_interval))
        for k in candidates
    )
    max_comm = max(max_comm, 1.0)

    for k in candidates:
        try:
            topology, metrics = build_mdfeel_topology(
                clients,
                k,
                feasible_graph,
                seed=seed,
                max_iter=max_iter,
            )
        except RuntimeError as exc:
            k_search.append({"K": k, "connected": False, "error": str(exc)})
            continue

        comm = communication_proxy(
            len(clients),
            k,
            topology.leader_edges,
            model_size=1.0,
            mixing_interval=max(1, mixing_interval),
        )
        score = float(metrics["similarity_cost"]) + lambda_communication * (comm / max_comm)
        row = {
            **metrics,
            "communication_proxy_unit_model": float(comm),
            "mdfeel_selection_score": score,
        }
        k_search.append(row)
        candidate = (score, k, topology, row)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None:
        raise RuntimeError("adapted MD-FEEL failed for every candidate K.")

    _, _, topology, metrics = best
    metrics = dict(metrics)
    metrics["k_search"] = k_search
    return topology, metrics


class MDFeelTrainingRunner(ClusteredAsyncTrainingRunner):
    """Adapted MD-FEEL runner using the shared clustered-async simulator."""


def _initial_centroids(
    clients: list[ClientMetadata],
    histograms: Mapping[int, np.ndarray],
    global_hist: np.ndarray,
    k: int,
) -> list[np.ndarray]:
    first = min(
        clients,
        key=lambda client: (_l1(histograms[client.client_id], global_hist), client.client_id),
    )
    selected = [first.client_id]
    while len(selected) < k:
        remaining = [client.client_id for client in clients if client.client_id not in selected]
        next_id = max(
            remaining,
            key=lambda cid: (
                min(_l1(histograms[cid], histograms[chosen]) for chosen in selected),
                -cid,
            ),
        )
        selected.append(next_id)
    return [histograms[cid].copy() for cid in selected]


def _assign_by_similarity(
    clients: list[ClientMetadata],
    histograms: Mapping[int, np.ndarray],
    centroids: list[np.ndarray],
    capacities: list[int],
) -> list[list[int]]:
    clusters: list[list[int]] = [[] for _ in capacities]
    remaining = list(capacities)

    def confidence(client: ClientMetadata) -> tuple[float, int]:
        distances = sorted(_l1(histograms[client.client_id], centroid) for centroid in centroids)
        gap = distances[1] - distances[0] if len(distances) > 1 else math.inf
        return (-gap, client.client_id)

    for client in sorted(clients, key=confidence):
        cid = client.client_id
        choices = [
            (_l1(histograms[cid], centroids[cluster_idx]), cluster_idx)
            for cluster_idx, slots in enumerate(remaining)
            if slots > 0
        ]
        if not choices:
            raise RuntimeError("capacity assignment exhausted before all clients were assigned.")
        _, cluster_idx = min(choices)
        clusters[cluster_idx].append(cid)
        remaining[cluster_idx] -= 1

    return [sorted(cluster) for cluster in clusters]


def _cluster_centroids(
    clusters: list[list[int]],
    clients_by_id: Mapping[int, ClientMetadata],
    histograms: Mapping[int, np.ndarray],
) -> list[np.ndarray]:
    centroids = []
    for cluster in clusters:
        vectors = [histograms[cid] for cid in cluster]
        weights = np.asarray([clients_by_id[cid].num_samples for cid in cluster], dtype=np.float64)
        if weights.sum() <= 0.0:
            centroids.append(np.mean(vectors, axis=0))
        else:
            centroids.append(np.average(np.stack(vectors), axis=0, weights=weights))
    return centroids


def _mdfeel_metrics(
    topology: Topology,
    clients_by_id: Mapping[int, ClientMetadata],
    histograms: Mapping[int, np.ndarray],
    centroids: list[np.ndarray],
    global_hist: np.ndarray,
) -> dict[str, Any]:
    weighted_distance = 0.0
    total_samples = 0.0
    cluster_distances = []
    for cluster_idx, cluster in enumerate(topology.clusters):
        distances = []
        for cid in cluster:
            weight = float(clients_by_id[cid].num_samples)
            distance = _l1(histograms[cid], centroids[cluster_idx])
            weighted_distance += weight * distance
            total_samples += weight
            distances.append(distance)
        cluster_distances.append(float(np.mean(distances)) if distances else 0.0)

    similarity_cost = weighted_distance / total_samples if total_samples > 0.0 else 0.0
    return {
        "similarity_cost": float(similarity_cost),
        "mean_cluster_similarity_cost": float(np.mean(cluster_distances)) if cluster_distances else 0.0,
        "max_cluster_similarity_cost": float(np.max(cluster_distances)) if cluster_distances else 0.0,
        "waiting_cost": waiting_cost(topology.clusters, clients_by_id),
        "distribution_cost": distribution_cost(topology.clusters, clients_by_id, global_hist),
        "cluster_sizes": [len(cluster) for cluster in topology.clusters],
    }


def _l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right).sum())
