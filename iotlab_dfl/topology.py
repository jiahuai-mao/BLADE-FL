from __future__ import annotations

import math
import time

import numpy as np


def sparse_regular_graph(client_ids, degree=6):
    ids = sorted(int(item) for item in client_ids)
    n = len(ids)
    if degree <= 0 or degree >= n or (n * degree) % 2:
        raise ValueError("degree must be positive, below N, and N*degree must be even")
    graph = {cid: set() for cid in ids}
    half = degree // 2
    for index, cid in enumerate(ids):
        for offset in range(1, half + 1):
            other = ids[(index + offset) % n]
            graph[cid].add(other)
            graph[other].add(cid)
    if degree % 2:
        if n % 2:
            raise ValueError("odd degree requires an even number of clients")
        for index in range(n // 2):
            left, right = ids[index], ids[index + n // 2]
            graph[left].add(right)
            graph[right].add(left)
    return graph


def random_connected_graph(client_ids, edge_probability, seed):
    ids = sorted(int(item) for item in client_ids)
    rng = np.random.RandomState(seed)
    graph = {cid: set() for cid in ids}
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            if rng.rand() < edge_probability:
                graph[left].add(right)
                graph[right].add(left)
    for index, left in enumerate(ids):
        right = ids[(index + 1) % len(ids)]
        graph[left].add(right)
        graph[right].add(left)
    return graph


def metropolis_weights(graph):
    weights = {}
    for node, neighbors in graph.items():
        row = {}
        for neighbor in neighbors:
            row[neighbor] = 1.0 / (1.0 + max(len(neighbors), len(graph[neighbor])))
        row[node] = 1.0 - sum(row.values())
        weights[node] = row
    return weights


def build_cluster_topology(clients, method, k_min=2, k_max=None, min_cluster_size=2, max_swaps=10, mixing_interval=1):
    started = time.time()
    n = len(clients)
    if k_max is None:
        k_max = max(k_min, int(math.floor(math.sqrt(n))))
    k_max = min(k_max, n // min_cluster_size)
    candidates = []
    built = []
    for k in range(k_min, k_max + 1):
        candidate_start = time.time()
        if method == "bladefl":
            clusters = _balanced_clusters(clients, k, max_swaps)
            wait, distribution = topology_costs(clients, clusters)
            base_score = 0.5 * wait + 0.5 * distribution
        elif method == "mdfeel":
            clusters = _similarity_clusters(clients, k)
            wait, distribution = topology_costs(clients, clusters)
            base_score = distribution
        else:
            raise ValueError("unsupported clustered method: %s" % method)
        leaders = [_leader(cluster, clients) for cluster in clusters]
        leader_edges = _ring_edges(leaders)
        communication = 2.0 * (n - k) + 2.0 * len(leader_edges) / float(max(mixing_interval, 1))
        built.append((k, clusters, leaders, leader_edges, wait, distribution, base_score, communication))
        candidates.append(
            {
                "K": k,
                "normalized_waiting_cost": wait,
                "normalized_distribution_cost": distribution,
                "balanced_score": 0.5 * wait + 0.5 * distribution,
                "communication_proxy_unit_model": communication,
                "construction_wall_time_sec": time.time() - candidate_start,
            }
        )
    communications = [item[7] for item in built]
    cmin, cmax = min(communications), max(communications)
    best = None
    for item, row in zip(built, candidates):
        normalized_comm = 0.0 if cmax == cmin else (item[7] - cmin) / (cmax - cmin)
        score = 0.8 * item[6] + 0.2 * normalized_comm
        row["normalized_communication_proxy"] = normalized_comm
        row["topology_score"] = score
        row["selected"] = False
        if best is None or (score, item[0]) < (best[0], best[1][0]):
            best = (score, item)
    if best is None:
        raise RuntimeError("no feasible clustered topology")
    score, selected = best
    selected_k, clusters, leaders, leader_edges, wait, distribution, _, communication = selected
    for row in candidates:
        row["selected"] = row["K"] == selected_k
    client_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for client_id in cluster:
            client_to_cluster[int(client_id)] = cluster_id
    return {
        "method": method,
        "K": selected_k,
        "clusters": clusters,
        "leaders": leaders,
        "leader_edges": leader_edges,
        "client_to_cluster": client_to_cluster,
        "normalized_waiting_cost": wait,
        "normalized_distribution_cost": distribution,
        "communication_proxy_unit_model": communication,
        "topology_score": score,
        "candidate_rows": candidates,
        "construction_wall_time_sec": time.time() - started,
    }


def topology_costs(clients, clusters):
    by_id = {int(item["client_id"]): item for item in clients}
    times = [float(item["profile_train_sec"]) for item in clients]
    denominator = len(clients) * max(max(times) - min(times), 1e-12)
    waiting = 0.0
    global_counts = np.sum([np.asarray(item["label_counts"], dtype=np.float64) for item in clients], axis=0)
    global_hist = global_counts / max(float(global_counts.sum()), 1.0)
    distribution = 0.0
    for cluster in clusters:
        maximum = max(float(by_id[cid]["profile_train_sec"]) for cid in cluster)
        waiting += sum(maximum - float(by_id[cid]["profile_train_sec"]) for cid in cluster)
        counts = np.sum([np.asarray(by_id[cid]["label_counts"], dtype=np.float64) for cid in cluster], axis=0)
        histogram = counts / max(float(counts.sum()), 1.0)
        distribution += float(np.abs(histogram - global_hist).sum())
    return waiting / denominator, distribution / (2.0 * len(clusters))


def _capacities(n, k):
    base, remainder = divmod(n, k)
    return [base + (1 if index < remainder else 0) for index in range(k)]


def _balanced_clusters(clients, k, max_swaps):
    capacities = _capacities(len(clients), k)
    by_id = {int(item["client_id"]): item for item in clients}
    global_counts = np.sum([np.asarray(item["label_counts"], dtype=np.float64) for item in clients], axis=0)
    global_hist = global_counts / global_counts.sum()
    ordered = sorted(
        by_id,
        key=lambda cid: (
            -float(np.abs(np.asarray(by_id[cid]["label_counts"], dtype=np.float64) / by_id[cid]["num_samples"] - global_hist).sum()),
            -float(by_id[cid]["profile_train_sec"]),
            cid,
        ),
    )
    clusters = [[] for _ in range(k)]
    for cid in ordered:
        choices = []
        for index, capacity in enumerate(capacities):
            if len(clusters[index]) >= capacity:
                continue
            trial = [list(cluster) for cluster in clusters]
            trial[index].append(cid)
            filled = [cluster for cluster in trial if cluster]
            wait, dist = topology_costs(clients, filled)
            choices.append((0.5 * wait + 0.5 * dist, index))
        clusters[min(choices)[1]].append(cid)
    current = sum(topology_costs(clients, clusters))
    for _ in range(max_swaps):
        improvement = None
        for left in range(k):
            for right in range(left + 1, k):
                for cid_left in clusters[left]:
                    for cid_right in clusters[right]:
                        trial = [list(cluster) for cluster in clusters]
                        il, ir = trial[left].index(cid_left), trial[right].index(cid_right)
                        trial[left][il], trial[right][ir] = cid_right, cid_left
                        score = sum(topology_costs(clients, trial))
                        if score + 1e-12 < current and (improvement is None or score < improvement[0]):
                            improvement = (score, trial)
        if improvement is None:
            break
        current, clusters = improvement
    return [sorted(cluster) for cluster in clusters]


def _similarity_clusters(clients, k):
    capacities = _capacities(len(clients), k)
    by_id = {int(item["client_id"]): item for item in clients}
    hist = {
        cid: np.asarray(item["label_counts"], dtype=np.float64) / max(float(item["num_samples"]), 1.0)
        for cid, item in by_id.items()
    }
    selected = [min(by_id)]
    while len(selected) < k:
        remaining = [cid for cid in by_id if cid not in selected]
        selected.append(max(remaining, key=lambda cid: min(float(np.abs(hist[cid] - hist[item]).sum()) for item in selected)))
    centroids = [hist[cid].copy() for cid in selected]
    clusters = [[] for _ in range(k)]
    for _ in range(8):
        clusters = [[] for _ in range(k)]
        remaining = list(capacities)
        for cid in sorted(by_id):
            choices = [(float(np.abs(hist[cid] - centroids[index]).sum()), index) for index in range(k) if remaining[index] > 0]
            index = min(choices)[1]
            clusters[index].append(cid)
            remaining[index] -= 1
        next_centroids = []
        for cluster in clusters:
            weights = np.asarray([by_id[cid]["num_samples"] for cid in cluster], dtype=np.float64)
            values = np.asarray([hist[cid] for cid in cluster], dtype=np.float64)
            next_centroids.append(np.average(values, axis=0, weights=weights))
        if all(np.allclose(left, right) for left, right in zip(centroids, next_centroids)):
            break
        centroids = next_centroids
    return [sorted(cluster) for cluster in clusters]


def _leader(cluster, clients):
    by_id = {int(item["client_id"]): item for item in clients}
    return min(cluster, key=lambda cid: (float(by_id[cid]["profile_train_sec"]), cid))


def _ring_edges(leaders):
    if len(leaders) <= 1:
        return []
    if len(leaders) == 2:
        return [[int(leaders[0]), int(leaders[1])]]
    return [[int(leaders[index]), int(leaders[(index + 1) % len(leaders)])] for index in range(len(leaders))]
