from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .topology import Graph, build_complete_graph, select_connected_leaders
from .types import ClientMetadata, Topology


@dataclass(frozen=True)
class ClusteringParams:
    k_min: int = 2
    k_max: int | None = None
    min_clients_per_cluster: int = 2
    alpha_time: float = 0.5
    alpha_distribution: float = 0.5
    lambda_communication: float = 0.2
    model_size: float = 1.0
    mixing_interval: int = 5
    epsilon: float = 1.0e-12
    epsilon_swap: float = 1.0e-12
    max_swaps: int = 10


def compute_capacities(num_clients: int, k: int) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > num_clients:
        raise ValueError(f"k={k} cannot exceed num_clients={num_clients}.")
    base = num_clients // k
    remainder = num_clients - k * base
    return [base + 1 if idx < remainder else base for idx in range(k)]


def candidate_k_values(num_clients: int, k_min: int, min_clients_per_cluster: int, k_max: int | None = None) -> list[int]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if k_min <= 0:
        raise ValueError("k_min must be positive.")
    if min_clients_per_cluster <= 0:
        raise ValueError("min_clients_per_cluster must be positive.")
    if k_max is not None and k_max <= 0:
        raise ValueError("k_max must be positive when provided.")
    upper = num_clients // min_clients_per_cluster
    if k_max is not None:
        upper = min(upper, k_max)
    if upper < k_min:
        return []
    return list(range(k_min, upper + 1))


def communication_proxy(
    num_clients: int,
    k: int,
    leader_edges: list[tuple[int, int]],
    model_size: float,
    mixing_interval: int,
) -> float:
    if model_size <= 0:
        raise ValueError("model_size must be positive.")
    if mixing_interval <= 0:
        raise ValueError("mixing_interval must be positive.")
    return float(2.0 * model_size * (num_clients - k) + (2.0 * model_size * len(leader_edges)) / mixing_interval)


def client_histogram(client: ClientMetadata) -> np.ndarray:
    total = float(client.num_samples)
    if total <= 0:
        return np.zeros(len(client.label_counts), dtype=np.float64)
    return np.asarray(client.label_counts, dtype=np.float64) / total


def global_histogram(clients: list[ClientMetadata]) -> np.ndarray:
    if not clients:
        raise ValueError("clients must not be empty.")
    counts = np.zeros(len(clients[0].label_counts), dtype=np.float64)
    total = 0.0
    for client in clients:
        counts += np.asarray(client.label_counts, dtype=np.float64)
        total += float(client.num_samples)
    if total <= 0:
        return counts
    return counts / total


def waiting_cost(clusters: list[list[int]], clients_by_id: Mapping[int, ClientMetadata]) -> float:
    cost = 0.0
    for cluster in clusters:
        if not cluster:
            continue
        max_time = max(clients_by_id[cid].train_time for cid in cluster)
        for cid in cluster:
            cost += max_time - clients_by_id[cid].train_time
    return float(cost)


def distribution_cost(
    clusters: list[list[int]],
    clients_by_id: Mapping[int, ClientMetadata],
    global_hist: np.ndarray,
) -> float:
    if not clusters:
        return 0.0
    cost = 0.0
    num_classes = len(global_hist)
    for cluster in clusters:
        counts = np.zeros(num_classes, dtype=np.float64)
        total = 0.0
        for cid in cluster:
            client = clients_by_id[cid]
            counts += np.asarray(client.label_counts, dtype=np.float64)
            total += float(client.num_samples)
        cluster_hist = counts if total <= 0 else counts / total
        cost += float(np.abs(cluster_hist - global_hist).sum())
    return cost / float(len(clusters))


def _cluster_waiting_cost(cluster: list[int], clients_by_id: Mapping[int, ClientMetadata]) -> float:
    if not cluster:
        return 0.0
    max_time = max(clients_by_id[cid].train_time for cid in cluster)
    return float(sum(max_time - clients_by_id[cid].train_time for cid in cluster))


def _cluster_distribution_distance(
    cluster: list[int],
    clients_by_id: Mapping[int, ClientMetadata],
    global_hist: np.ndarray,
) -> float:
    counts = np.zeros(len(global_hist), dtype=np.float64)
    total = 0.0
    for cid in cluster:
        client = clients_by_id[cid]
        counts += np.asarray(client.label_counts, dtype=np.float64)
        total += float(client.num_samples)
    cluster_hist = counts if total <= 0 else counts / total
    return float(np.abs(cluster_hist - global_hist).sum())


def _distribution_distance_from_counts(
    counts: np.ndarray,
    sample_total: float,
    global_hist: np.ndarray,
) -> float:
    cluster_hist = counts if sample_total <= 0 else counts / sample_total
    return float(np.abs(cluster_hist - global_hist).sum())


def balanced_score(
    clusters: list[list[int]],
    clients_by_id: Mapping[int, ClientMetadata],
    global_hist: np.ndarray,
    alpha_time: float,
    alpha_distribution: float,
    epsilon: float = 1.0e-12,
) -> tuple[float, dict]:
    clients = list(clients_by_id.values())
    train_times = [client.train_time for client in clients]
    raw_waiting = waiting_cost(clusters, clients_by_id)
    raw_distribution = distribution_cost(clusters, clients_by_id, global_hist)
    normalized_waiting = raw_waiting / (len(clients) * (max(train_times) - min(train_times)) + epsilon)
    normalized_distribution = raw_distribution / 2.0
    score = alpha_time * normalized_waiting + alpha_distribution * normalized_distribution
    return float(score), {
        "waiting_cost": float(raw_waiting),
        "distribution_cost": float(raw_distribution),
        "normalized_waiting_cost": float(normalized_waiting),
        "normalized_distribution_cost": float(normalized_distribution),
        "balanced_score": float(score),
    }


def _validate_inputs(clients: list[ClientMetadata], k: int, alpha_time: float, alpha_distribution: float) -> None:
    if not clients:
        raise ValueError("clients must not be empty.")
    if k <= 0 or k > len(clients):
        raise ValueError(f"k must be in [1, {len(clients)}], got {k}.")
    if any(client.num_samples < 0 for client in clients):
        raise ValueError("num_samples must be non-negative.")
    if any(client.train_time < 0 for client in clients):
        raise ValueError("train_time must be non-negative.")
    num_classes = len(clients[0].label_counts)
    if num_classes == 0:
        raise ValueError("label_counts must not be empty.")
    if any(len(client.label_counts) != num_classes for client in clients):
        raise ValueError("all clients must have the same number of label counts.")
    if any(sum(client.label_counts) != client.num_samples for client in clients):
        raise ValueError("sum(label_counts) must equal num_samples for each client.")
    if alpha_time < 0 or alpha_distribution < 0:
        raise ValueError("alpha weights must be non-negative.")
    if not math.isclose(alpha_time + alpha_distribution, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError("alpha_time + alpha_distribution must equal 1.")


def _ordered_client_ids(clients: list[ClientMetadata], global_hist: np.ndarray) -> list[int]:
    distances = {}
    for client in clients:
        distances[client.client_id] = float(np.abs(client_histogram(client) - global_hist).sum())
    return [
        client.client_id
        for client in sorted(
            clients,
            key=lambda c: (-distances[c.client_id], -c.train_time, c.client_id),
        )
    ]


def _greedy_initial_partition(
    ordered_ids: list[int],
    capacities: list[int],
    clients_by_id: Mapping[int, ClientMetadata],
    global_hist: np.ndarray,
    alpha_time: float,
    alpha_distribution: float,
    epsilon: float,
) -> list[list[int]]:
    k = len(capacities)
    clusters = [[] for _ in range(k)]

    for cluster_idx, cid in enumerate(ordered_ids[:k]):
        if capacities[cluster_idx] < 1:
            raise ValueError(f"cluster {cluster_idx} has zero capacity.")
        clusters[cluster_idx].append(cid)

    clients = list(clients_by_id.values())
    train_times = [client.train_time for client in clients]
    waiting_den = len(clients) * (max(train_times) - min(train_times)) + epsilon
    cluster_waiting = [_cluster_waiting_cost(cluster, clients_by_id) for cluster in clusters]
    client_counts = {
        cid: np.asarray(client.label_counts, dtype=np.float64)
        for cid, client in clients_by_id.items()
    }
    client_samples = {
        cid: float(client.num_samples)
        for cid, client in clients_by_id.items()
    }
    cluster_counts = []
    cluster_sample_totals = []
    cluster_distribution = []
    for cluster in clusters:
        counts = np.zeros(len(global_hist), dtype=np.float64)
        sample_total = 0.0
        for cid in cluster:
            counts += client_counts[cid]
            sample_total += client_samples[cid]
        cluster_counts.append(counts)
        cluster_sample_totals.append(sample_total)
        cluster_distribution.append(_distribution_distance_from_counts(counts, sample_total, global_hist))

    for cid in ordered_ids[k:]:
        best_cluster = None
        best_delta = float("inf")
        best_waiting = 0.0
        best_distribution = 0.0

        for cluster_idx in range(k):
            if len(clusters[cluster_idx]) >= capacities[cluster_idx]:
                continue

            proposal_cluster = [*clusters[cluster_idx], cid]
            new_waiting = _cluster_waiting_cost(proposal_cluster, clients_by_id)
            new_counts = cluster_counts[cluster_idx] + client_counts[cid]
            new_sample_total = cluster_sample_totals[cluster_idx] + client_samples[cid]
            new_distribution = _distribution_distance_from_counts(new_counts, new_sample_total, global_hist)
            delta = (
                alpha_time * ((new_waiting - cluster_waiting[cluster_idx]) / waiting_den)
                + alpha_distribution * (((new_distribution - cluster_distribution[cluster_idx]) / k) / 2.0)
            )
            if best_cluster is None or delta < best_delta:
                best_cluster = cluster_idx
                best_delta = delta
                best_waiting = new_waiting
                best_distribution = new_distribution

        if best_cluster is None:
            raise RuntimeError("no cluster has remaining capacity during greedy insertion.")
        clusters[best_cluster].append(cid)
        cluster_waiting[best_cluster] = best_waiting
        cluster_distribution[best_cluster] = best_distribution
        cluster_counts[best_cluster] = cluster_counts[best_cluster] + client_counts[cid]
        cluster_sample_totals[best_cluster] += client_samples[cid]

    return clusters


def _refine_by_swaps(
    clusters: list[list[int]],
    clients_by_id: Mapping[int, ClientMetadata],
    global_hist: np.ndarray,
    alpha_time: float,
    alpha_distribution: float,
    epsilon: float,
    epsilon_swap: float,
    max_swaps: int,
) -> tuple[list[list[int]], int]:
    clusters = [sorted(cluster) for cluster in clusters]
    accepted = 0
    num_clusters = len(clusters)
    clients = list(clients_by_id.values())
    train_times = [client.train_time for client in clients]
    waiting_den = len(clients) * (max(train_times) - min(train_times)) + epsilon

    client_counts = {
        cid: np.asarray(client.label_counts, dtype=np.float64)
        for cid, client in clients_by_id.items()
    }
    client_samples = {
        cid: float(client.num_samples)
        for cid, client in clients_by_id.items()
    }
    cluster_waiting = [_cluster_waiting_cost(cluster, clients_by_id) for cluster in clusters]
    cluster_counts = []
    cluster_sample_totals = []
    cluster_distribution = []
    for cluster in clusters:
        counts = np.zeros(len(global_hist), dtype=np.float64)
        sample_total = 0.0
        for cid in cluster:
            counts += client_counts[cid]
            sample_total += client_samples[cid]
        cluster_counts.append(counts)
        cluster_sample_totals.append(sample_total)
        cluster_distribution.append(_distribution_distance_from_counts(counts, sample_total, global_hist))
    current_score = (
        alpha_time * (sum(cluster_waiting) / waiting_den)
        + alpha_distribution * ((sum(cluster_distribution) / num_clusters) / 2.0)
    )

    while accepted < max_swaps:
        best_decrease = 0.0
        best_updates: tuple[int, int, float, float, float, float] | None = None
        best_clusters: tuple[list[int], list[int]] | None = None
        best_count_updates: tuple[np.ndarray, np.ndarray, float, float] | None = None

        for k1 in range(num_clusters):
            for k2 in range(k1 + 1, num_clusters):
                for cid1 in sorted(clusters[k1]):
                    for cid2 in sorted(clusters[k2]):
                        new_cluster1 = [cid2 if cid == cid1 else cid for cid in clusters[k1]]
                        new_cluster2 = [cid1 if cid == cid2 else cid for cid in clusters[k2]]
                        new_cluster1.sort()
                        new_cluster2.sort()

                        new_wait1 = _cluster_waiting_cost(new_cluster1, clients_by_id)
                        new_wait2 = _cluster_waiting_cost(new_cluster2, clients_by_id)
                        new_counts1 = cluster_counts[k1] - client_counts[cid1] + client_counts[cid2]
                        new_counts2 = cluster_counts[k2] - client_counts[cid2] + client_counts[cid1]
                        new_total1 = cluster_sample_totals[k1] - client_samples[cid1] + client_samples[cid2]
                        new_total2 = cluster_sample_totals[k2] - client_samples[cid2] + client_samples[cid1]
                        new_dist1 = _distribution_distance_from_counts(new_counts1, new_total1, global_hist)
                        new_dist2 = _distribution_distance_from_counts(new_counts2, new_total2, global_hist)

                        waiting_delta = (
                            new_wait1
                            + new_wait2
                            - cluster_waiting[k1]
                            - cluster_waiting[k2]
                        )
                        distribution_delta = (
                            new_dist1
                            + new_dist2
                            - cluster_distribution[k1]
                            - cluster_distribution[k2]
                        )
                        proposal_score = current_score + (
                            alpha_time * (waiting_delta / waiting_den)
                            + alpha_distribution * ((distribution_delta / num_clusters) / 2.0)
                        )
                        decrease = current_score - proposal_score
                        if decrease > best_decrease + epsilon:
                            best_decrease = decrease
                            best_updates = (k1, k2, new_wait1, new_wait2, new_dist1, new_dist2)
                            best_clusters = (new_cluster1, new_cluster2)
                            best_count_updates = (new_counts1, new_counts2, new_total1, new_total2)

        if best_clusters is None or best_updates is None or best_count_updates is None or best_decrease <= epsilon_swap:
            break
        k1, k2, new_wait1, new_wait2, new_dist1, new_dist2 = best_updates
        new_counts1, new_counts2, new_total1, new_total2 = best_count_updates
        clusters[k1], clusters[k2] = best_clusters
        cluster_waiting[k1] = new_wait1
        cluster_waiting[k2] = new_wait2
        cluster_distribution[k1] = new_dist1
        cluster_distribution[k2] = new_dist2
        cluster_counts[k1] = new_counts1
        cluster_counts[k2] = new_counts2
        cluster_sample_totals[k1] = new_total1
        cluster_sample_totals[k2] = new_total2
        current_score -= best_decrease
        accepted += 1

    return clusters, accepted


def _client_to_cluster(clusters: list[list[int]]) -> dict[int, int]:
    mapping = {}
    for cluster_idx, cluster in enumerate(clusters):
        for cid in cluster:
            mapping[cid] = cluster_idx
    return mapping


def _validate_topology(clusters: list[list[int]], capacities: list[int], client_ids: list[int]) -> None:
    if [len(cluster) for cluster in clusters] != capacities:
        raise RuntimeError("cluster sizes do not match fixed capacities.")
    flattened = [cid for cluster in clusters for cid in cluster]
    if sorted(flattened) != sorted(client_ids):
        raise RuntimeError("each client must appear exactly once in clusters.")


def build_deterministic_balanced_clusters(
    clients: list[ClientMetadata],
    k: int,
    alpha_time: float = 0.5,
    alpha_distribution: float = 0.5,
    epsilon: float = 1.0e-12,
    epsilon_swap: float = 1.0e-12,
    max_swaps: int = 10,
    feasible_graph: Graph | None = None,
) -> tuple[Topology, dict]:
    """Build a deterministic capacity-constrained topology for fixed K."""

    _validate_inputs(clients, k, alpha_time, alpha_distribution)
    clients_by_id = {client.client_id: client for client in clients}
    client_ids = sorted(clients_by_id)
    capacities = compute_capacities(len(clients), k)
    global_hist = global_histogram(clients)
    ordered_ids = _ordered_client_ids(clients, global_hist)

    clusters = _greedy_initial_partition(
        ordered_ids,
        capacities,
        clients_by_id,
        global_hist,
        alpha_time,
        alpha_distribution,
        epsilon,
    )
    clusters, accepted_swaps = _refine_by_swaps(
        clusters,
        clients_by_id,
        global_hist,
        alpha_time,
        alpha_distribution,
        epsilon,
        epsilon_swap,
        max_swaps,
    )
    clusters = [sorted(cluster) for cluster in clusters]
    _validate_topology(clusters, capacities, client_ids)

    graph = feasible_graph if feasible_graph is not None else build_complete_graph(client_ids)
    leaders, leader_edges, leader_score, connected = select_connected_leaders(clusters, clients_by_id, graph)
    if not connected:
        raise RuntimeError("no connected leader overlay can be formed for this partition.")

    score, score_parts = balanced_score(
        clusters,
        clients_by_id,
        global_hist,
        alpha_time,
        alpha_distribution,
        epsilon,
    )
    topology = Topology(
        clusters=clusters,
        leaders=leaders,
        leader_edges=leader_edges,
        client_to_cluster=_client_to_cluster(clusters),
    )
    metrics = {
        "K": k,
        "capacities": capacities,
        "accepted_swaps": accepted_swaps,
        "leader_score": float(leader_score),
        "leader_edges": leader_edges,
        "connected": connected,
        "global_histogram": global_hist.tolist(),
        **score_parts,
    }
    metrics["balanced_score"] = score
    return topology, metrics


def select_best_k_topology(
    clients: list[ClientMetadata],
    k_min: int = 2,
    k_max: int | None = None,
    min_clients_per_cluster: int = 2,
    alpha_time: float = 0.5,
    alpha_distribution: float = 0.5,
    lambda_communication: float = 0.2,
    model_size: float = 1.0,
    mixing_interval: int = 5,
    feasible_graph: Graph | None = None,
    epsilon: float = 1.0e-12,
    epsilon_swap: float = 1.0e-12,
    max_swaps: int = 10,
) -> tuple[Topology, dict]:
    if not 0 <= lambda_communication <= 1:
        raise ValueError("lambda_communication must be in [0, 1].")

    candidates = candidate_k_values(len(clients), k_min, min_clients_per_cluster, k_max)
    if not candidates:
        raise ValueError("No candidate K values for the active clients.")

    rows = []
    feasible: list[tuple[int, Topology, dict, float]] = []
    for k in candidates:
        candidate_start = time.perf_counter()
        try:
            topology, metrics = build_deterministic_balanced_clusters(
                clients,
                k=k,
                alpha_time=alpha_time,
                alpha_distribution=alpha_distribution,
                epsilon=epsilon,
                epsilon_swap=epsilon_swap,
                max_swaps=max_swaps,
                feasible_graph=feasible_graph,
            )
            candidate_wall_time = time.perf_counter() - candidate_start
            proxy = communication_proxy(
                num_clients=len(clients),
                k=k,
                leader_edges=topology.leader_edges,
                model_size=model_size,
                mixing_interval=mixing_interval,
            )
            row = {
                "K": k,
                "feasible": True,
                "capacities": metrics["capacities"],
                "balanced_score": metrics["balanced_score"],
                "waiting_cost": metrics["waiting_cost"],
                "distribution_cost": metrics["distribution_cost"],
                "normalized_waiting_cost": metrics["normalized_waiting_cost"],
                "normalized_distribution_cost": metrics["normalized_distribution_cost"],
                "leader_score": metrics["leader_score"],
                "leaders": topology.leaders,
                "leader_edges": topology.leader_edges,
                "communication_proxy": proxy,
                "normalized_communication_proxy": None,
                "topology_score": None,
                "selected": False,
                "failure_reason": None,
                "construction_wall_time_sec": float(candidate_wall_time),
            }
            rows.append(row)
            feasible.append((k, topology, metrics, proxy))
        except Exception as exc:
            candidate_wall_time = time.perf_counter() - candidate_start
            rows.append(
                {
                    "K": k,
                    "feasible": False,
                    "capacities": compute_capacities(len(clients), k),
                    "balanced_score": None,
                    "waiting_cost": None,
                    "distribution_cost": None,
                    "normalized_waiting_cost": None,
                    "normalized_distribution_cost": None,
                    "leader_score": None,
                    "leaders": [],
                    "leader_edges": [],
                    "communication_proxy": None,
                    "normalized_communication_proxy": None,
                    "topology_score": None,
                    "selected": False,
                    "failure_reason": str(exc),
                    "construction_wall_time_sec": float(candidate_wall_time),
                }
            )

    if not feasible:
        raise RuntimeError("No feasible topology found for active clients.")

    proxies = [item[3] for item in feasible]
    proxy_min = min(proxies)
    proxy_max = max(proxies)
    best_k = None
    best_topology = None
    best_metrics = None
    best_score = float("inf")

    for k, topology, metrics, proxy in feasible:
        if proxy_max == proxy_min:
            normalized_proxy = 0.0
        else:
            normalized_proxy = (proxy - proxy_min) / (proxy_max - proxy_min + epsilon)
        topology_score = (1.0 - lambda_communication) * metrics["balanced_score"] + lambda_communication * normalized_proxy

        for row in rows:
            if row["K"] == k:
                row["normalized_communication_proxy"] = float(normalized_proxy)
                row["topology_score"] = float(topology_score)
                break

        if best_k is None or topology_score < best_score or (topology_score == best_score and k < best_k):
            best_k = k
            best_topology = topology
            best_metrics = metrics
            best_score = topology_score

    if best_k is None or best_topology is None or best_metrics is None:
        raise RuntimeError("No feasible topology found for active clients.")

    for row in rows:
        row["selected"] = row["K"] == best_k

    search = {
        "selected_k": best_k,
        "candidate_rows": rows,
        "lambda_communication": float(lambda_communication),
        "model_size": float(model_size),
        "mixing_interval": int(mixing_interval),
    }
    metrics = dict(best_metrics)
    metrics["topology_score"] = float(best_score)
    metrics["normalized_communication_proxy"] = next(
        row["normalized_communication_proxy"] for row in rows if row["K"] == best_k
    )
    metrics["communication_proxy"] = next(row["communication_proxy"] for row in rows if row["K"] == best_k)
    metrics["k_search"] = search
    return best_topology, metrics


def refresh_topology(
    previous_topology: Topology | None,
    previous_clients: list[ClientMetadata],
    updated_clients: list[ClientMetadata],
    previous_cluster_models: list[Any] | None = None,
    updated_feasible_graph: Graph | None = None,
    params: ClusteringParams | None = None,
    refresh_round: int = 0,
    departed_clients: set[int] | None = None,
    newly_joined_clients: set[int] | None = None,
    removed_unavailable_clients: set[int] | None = None,
) -> tuple[Topology | None, dict]:
    """Repair or reconstruct a topology according to Supplementary Method 3.

    The returned log is JSON-friendly except for caller-owned model objects, which are
    intentionally not stored in it. Use ``compute_model_transfer_weights`` and
    ``transfer_cluster_models`` when model warm-start vectors are needed.
    """

    params = params or ClusteringParams()
    if not updated_clients:
        raise ValueError("updated_clients must not be empty.")
    if previous_cluster_models is not None and previous_topology is None:
        raise ValueError("previous_cluster_models requires previous_topology.")
    if previous_cluster_models is not None and len(previous_cluster_models) != len(previous_topology.clusters):
        raise ValueError("previous_cluster_models length must match previous topology K.")

    updated_ids = {client.client_id for client in updated_clients}
    previous_ids = {client.client_id for client in previous_clients}
    departed = set(departed_clients or (previous_ids - updated_ids))
    joined = set(newly_joined_clients or (updated_ids - previous_ids))
    removed_unavailable = set(removed_unavailable_clients or set())
    graph = updated_feasible_graph if updated_feasible_graph is not None else build_complete_graph(sorted(updated_ids))

    log = {
        "refresh_round": int(refresh_round),
        "previous_k": len(previous_topology.clusters) if previous_topology is not None else None,
        "new_k": None,
        "repair_attempted": False,
        "repair_succeeded": False,
        "reconstruction_used": False,
        "departed_clients": sorted(departed),
        "newly_joined_clients": sorted(joined),
        "removed_unavailable_clients": sorted(removed_unavailable),
        "capacity_violation_before_repair": {},
        "capacity_violation_after_repair": {},
        "connected_overlay_available": False,
        "mixing_suspended": False,
        "paused_clusters": [],
        "leader_edges": [],
        "model_transfer_weights": {},
        "transferred_cluster_model_count": 0,
        "failure_reason": None,
        "refresh_mode": None,
    }

    previous_k = len(previous_topology.clusters) if previous_topology is not None else None
    if previous_topology is not None and previous_k is not None and _is_k_refresh_feasible(
        previous_k,
        len(updated_clients),
        params.k_min,
        params.k_max,
        params.min_clients_per_cluster,
    ):
        log["repair_attempted"] = True
        repaired, repair_metrics, repair_reason = _repair_previous_topology(
            previous_topology=previous_topology,
            updated_clients=updated_clients,
            updated_feasible_graph=graph,
            params=params,
            joined_client_ids=joined,
            removed_client_ids=departed | removed_unavailable | (previous_ids - updated_ids),
        )
        log["capacity_violation_before_repair"] = repair_metrics.get("capacity_violation_before_repair", {})
        log["capacity_violation_after_repair"] = repair_metrics.get("capacity_violation_after_repair", {})
        if repaired is not None:
            log.update(_successful_refresh_log("repair", repaired, repair_metrics))
            if previous_topology is not None:
                log["model_transfer_weights"] = compute_model_transfer_weights(
                    previous_topology,
                    repaired,
                    {client.client_id: client for client in updated_clients},
                )
                if previous_cluster_models is not None:
                    log["transferred_cluster_model_count"] = len(
                        transfer_cluster_models(previous_cluster_models, log["model_transfer_weights"])
                    )
            return repaired, log
        log["failure_reason"] = repair_reason

    try:
        topology, metrics = select_best_k_topology(
            updated_clients,
            k_min=params.k_min,
            k_max=params.k_max,
            min_clients_per_cluster=params.min_clients_per_cluster,
            alpha_time=params.alpha_time,
            alpha_distribution=params.alpha_distribution,
            lambda_communication=params.lambda_communication,
            model_size=params.model_size,
            mixing_interval=params.mixing_interval,
            feasible_graph=graph,
            epsilon=params.epsilon,
            epsilon_swap=params.epsilon_swap,
            max_swaps=params.max_swaps,
        )
    except Exception as exc:
        log["reconstruction_used"] = True
        log["mixing_suspended"] = True
        log["connected_overlay_available"] = False
        log["failure_reason"] = str(exc)
        if previous_topology is not None:
            log["paused_clusters"] = _paused_clusters_without_available_leader(previous_topology, updated_ids)
        return None, log

    log.update(_successful_refresh_log("reconstruction", topology, metrics))
    log["reconstruction_used"] = True
    if previous_topology is not None:
        log["model_transfer_weights"] = compute_model_transfer_weights(
            previous_topology,
            topology,
            {client.client_id: client for client in updated_clients},
        )
        if previous_cluster_models is not None:
            log["transferred_cluster_model_count"] = len(
                transfer_cluster_models(previous_cluster_models, log["model_transfer_weights"])
            )
    return topology, log


def compute_model_transfer_weights(
    previous_topology: Topology,
    new_topology: Topology,
    clients_by_id: Mapping[int, ClientMetadata],
) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {}
    previous_members = [set(cluster) for cluster in previous_topology.clusters]
    previous_active = set().union(*previous_members) if previous_members else set()

    for new_idx, new_cluster in enumerate(new_topology.clusters):
        retained = [cid for cid in new_cluster if cid in previous_active and cid in clients_by_id]
        retained_samples = sum(float(clients_by_id[cid].num_samples) for cid in retained)
        row: dict[str, float] = {}
        if retained_samples > 0.0:
            for old_idx, old_members in enumerate(previous_members):
                overlap_samples = sum(
                    float(clients_by_id[cid].num_samples)
                    for cid in new_cluster
                    if cid in old_members and cid in clients_by_id
                )
                if overlap_samples > 0.0:
                    row[str(old_idx)] = float(overlap_samples / retained_samples)
        else:
            previous_k = len(previous_topology.clusters)
            if previous_k > 0:
                row = {str(old_idx): float(1.0 / previous_k) for old_idx in range(previous_k)}
        weights[str(new_idx)] = row
    return weights


def transfer_cluster_models(previous_cluster_models: list[Any], transfer_weights: Mapping[str, Mapping[str, float]]) -> list[Any]:
    new_models = []
    for new_idx in sorted(transfer_weights, key=lambda item: int(item)):
        weighted_model = None
        for old_idx, weight in sorted(transfer_weights[new_idx].items(), key=lambda item: int(item[0])):
            contribution = previous_cluster_models[int(old_idx)] * float(weight)
            weighted_model = contribution if weighted_model is None else weighted_model + contribution
        new_models.append(weighted_model)
    return new_models


def _repair_previous_topology(
    previous_topology: Topology,
    updated_clients: list[ClientMetadata],
    updated_feasible_graph: Graph,
    params: ClusteringParams,
    joined_client_ids: set[int],
    removed_client_ids: set[int],
) -> tuple[Topology | None, dict, str | None]:
    k = len(previous_topology.clusters)
    clients_by_id = {client.client_id: client for client in updated_clients}
    updated_ids = set(clients_by_id)
    capacities = compute_capacities(len(updated_clients), k)
    global_hist = global_histogram(updated_clients)
    clusters = [
        sorted(cid for cid in cluster if cid in updated_ids and cid not in removed_client_ids)
        for cluster in previous_topology.clusters
    ]

    metrics = {
        "K": k,
        "capacities": capacities,
        "capacity_violation_before_repair": _capacity_violation(clusters, capacities),
        "capacity_violation_after_repair": None,
    }

    assigned = {cid for cluster in clusters for cid in cluster}
    missing = (updated_ids - assigned) | (joined_client_ids & updated_ids)
    missing = {cid for cid in missing if cid not in assigned}
    ok = _insert_clients_for_repair(
        clusters,
        sorted(missing),
        capacities,
        clients_by_id,
        global_hist,
        params,
    )
    if not ok:
        metrics["capacity_violation_after_repair"] = _capacity_violation(clusters, capacities)
        return None, metrics, "repair failed while inserting newly joined clients."

    ok = _repair_capacity_by_moves(clusters, capacities, clients_by_id, global_hist, params)
    if not ok:
        metrics["capacity_violation_after_repair"] = _capacity_violation(clusters, capacities)
        return None, metrics, "repair failed while restoring target capacities."

    metrics["capacity_violation_after_repair"] = _capacity_violation(clusters, capacities)
    if [len(cluster) for cluster in clusters] != capacities:
        return None, metrics, "repair did not restore exact target capacities."

    clusters, accepted_swaps = _refine_by_swaps(
        clusters,
        clients_by_id,
        global_hist,
        params.alpha_time,
        params.alpha_distribution,
        params.epsilon,
        params.epsilon_swap,
        params.max_swaps,
    )
    clusters = [sorted(cluster) for cluster in clusters]
    try:
        _validate_topology(clusters, capacities, sorted(updated_ids))
    except RuntimeError as exc:
        return None, metrics, str(exc)

    leaders, leader_edges, leader_score, connected = select_connected_leaders(clusters, clients_by_id, updated_feasible_graph)
    if not connected:
        return None, metrics, "repair failed because no connected leader overlay can be formed."

    score, score_parts = balanced_score(
        clusters,
        clients_by_id,
        global_hist,
        params.alpha_time,
        params.alpha_distribution,
        params.epsilon,
    )
    topology = Topology(
        clusters=clusters,
        leaders=leaders,
        leader_edges=leader_edges,
        client_to_cluster=_client_to_cluster(clusters),
    )
    metrics.update(
        {
            "accepted_swaps": accepted_swaps,
            "leader_score": float(leader_score),
            "leader_edges": leader_edges,
            "connected": connected,
            "global_histogram": global_hist.tolist(),
            **score_parts,
        }
    )
    metrics["balanced_score"] = score
    return topology, metrics, None


def _insert_clients_for_repair(
    clusters: list[list[int]],
    client_ids: list[int],
    capacities: list[int],
    clients_by_id: Mapping[int, ClientMetadata],
    global_hist: np.ndarray,
    params: ClusteringParams,
) -> bool:
    if not client_ids:
        return True
    ordered = _ordered_client_ids([clients_by_id[cid] for cid in client_ids], global_hist)
    for cid in ordered:
        current_score, _ = balanced_score(
            clusters,
            clients_by_id,
            global_hist,
            params.alpha_time,
            params.alpha_distribution,
            params.epsilon,
        )
        best_cluster = None
        best_delta = float("inf")
        for cluster_idx in range(len(clusters)):
            if len(clusters[cluster_idx]) >= capacities[cluster_idx]:
                continue
            proposal = [list(cluster) for cluster in clusters]
            proposal[cluster_idx] = sorted([*proposal[cluster_idx], cid])
            proposal_score, _ = balanced_score(
                proposal,
                clients_by_id,
                global_hist,
                params.alpha_time,
                params.alpha_distribution,
                params.epsilon,
            )
            delta = proposal_score - current_score
            if best_cluster is None or delta < best_delta:
                best_cluster = cluster_idx
                best_delta = delta
        if best_cluster is None:
            return False
        clusters[best_cluster].append(cid)
        clusters[best_cluster].sort()
    return True


def _repair_capacity_by_moves(
    clusters: list[list[int]],
    capacities: list[int],
    clients_by_id: Mapping[int, ClientMetadata],
    global_hist: np.ndarray,
    params: ClusteringParams,
) -> bool:
    while [len(cluster) for cluster in clusters] != capacities:
        over = [idx for idx, cluster in enumerate(clusters) if len(cluster) > capacities[idx]]
        under = [idx for idx, cluster in enumerate(clusters) if len(cluster) < capacities[idx]]
        if not over or not under:
            return False

        current_score, _ = balanced_score(
            clusters,
            clients_by_id,
            global_hist,
            params.alpha_time,
            params.alpha_distribution,
            params.epsilon,
        )
        best_move: tuple[int, int, int] | None = None
        best_increase = float("inf")
        for src in over:
            for dst in under:
                for cid in sorted(clusters[src]):
                    proposal = [list(cluster) for cluster in clusters]
                    proposal[src].remove(cid)
                    proposal[dst].append(cid)
                    proposal[dst].sort()
                    proposal_score, _ = balanced_score(
                        proposal,
                        clients_by_id,
                        global_hist,
                        params.alpha_time,
                        params.alpha_distribution,
                        params.epsilon,
                    )
                    increase = proposal_score - current_score
                    if best_move is None or increase < best_increase:
                        best_move = (src, dst, cid)
                        best_increase = increase
        if best_move is None:
            return False
        src, dst, cid = best_move
        clusters[src].remove(cid)
        clusters[dst].append(cid)
        clusters[dst].sort()
    return True


def _is_k_refresh_feasible(k: int, num_clients: int, k_min: int, k_max: int | None, min_clients_per_cluster: int) -> bool:
    if num_clients <= 0:
        return False
    if k_max is not None and k > k_max:
        return False
    return k_min <= k <= num_clients and k <= num_clients // min_clients_per_cluster


def _capacity_violation(clusters: list[list[int]], capacities: list[int]) -> dict:
    rows = {}
    total_excess = 0
    total_deficit = 0
    for idx, cluster in enumerate(clusters):
        size = len(cluster)
        capacity = capacities[idx]
        excess = max(size - capacity, 0)
        deficit = max(capacity - size, 0)
        total_excess += excess
        total_deficit += deficit
        rows[str(idx)] = {
            "size": size,
            "capacity": capacity,
            "excess": excess,
            "deficit": deficit,
        }
    return {
        "clusters": rows,
        "total_excess": total_excess,
        "total_deficit": total_deficit,
    }


def _successful_refresh_log(mode: str, topology: Topology, metrics: dict) -> dict:
    return {
        "new_k": len(topology.clusters),
        "repair_succeeded": mode == "repair",
        "reconstruction_used": mode == "reconstruction",
        "connected_overlay_available": True,
        "mixing_suspended": False,
        "paused_clusters": [],
        "leader_edges": topology.leader_edges,
        "failure_reason": None,
        "refresh_mode": mode,
        "metrics": metrics,
    }


def _paused_clusters_without_available_leader(topology: Topology, active_ids: set[int]) -> list[int]:
    paused = []
    for cluster_idx, cluster in enumerate(topology.clusters):
        if not any(cid in active_ids for cid in cluster):
            paused.append(cluster_idx)
    return paused
