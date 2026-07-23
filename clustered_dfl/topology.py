from __future__ import annotations

from itertools import product
from typing import Mapping

from .types import ClientMetadata

Graph = Mapping[int, set[int]]


def build_complete_graph(client_ids: list[int]) -> dict[int, set[int]]:
    return {cid: {other for other in client_ids if other != cid} for cid in client_ids}


def is_connected_subset(nodes: list[int], graph: Graph) -> bool:
    if len(nodes) <= 1:
        return True

    node_set = set(nodes)
    start = nodes[0]
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neigh in graph.get(current, set()):
            if neigh in node_set and neigh not in seen:
                seen.add(neigh)
                stack.append(neigh)
    return len(seen) == len(node_set)


def is_complete_for_nodes(nodes: list[int], graph: Graph) -> bool:
    node_set = set(nodes)
    for node in nodes:
        expected = node_set - {node}
        if not expected.issubset(graph.get(node, set())):
            return False
    return True


def induced_leader_edges(leaders: list[int], graph: Graph) -> list[tuple[int, int]]:
    leader_set = set(leaders)
    edges = set()
    for src in leaders:
        for dst in graph.get(src, set()):
            if dst not in leader_set or src == dst:
                continue
            left, right = sorted((src, dst))
            edges.add((left, right))
    return sorted(edges)


def select_connected_leaders(
    clusters: list[list[int]],
    clients_by_id: Mapping[int, ClientMetadata],
    feasible_graph: Graph,
) -> tuple[list[int], list[tuple[int, int]], float, bool]:
    """Select one leader per cluster according to Supplementary Method 2."""

    candidate_lists = []
    for cluster in clusters:
        if not cluster:
            return [], [], float("inf"), False
        candidates = sorted(
            cluster,
            key=lambda cid: (clients_by_id[cid].train_time, cid),
        )
        candidate_lists.append(candidates)

    all_client_ids = sorted(clients_by_id)
    if is_complete_for_nodes(all_client_ids, feasible_graph):
        best_leaders = [candidates[0] for candidates in candidate_lists]
        best_score = sum(clients_by_id[cid].train_time for cid in best_leaders)
        return best_leaders, induced_leader_edges(best_leaders, feasible_graph), best_score, True

    best_leaders: list[int] | None = None
    best_score = float("inf")

    for combo in product(*candidate_lists):
        leaders = list(combo)
        if not is_connected_subset(leaders, feasible_graph):
            continue
        score = sum(clients_by_id[cid].train_time for cid in leaders)
        if best_leaders is None:
            best_leaders = leaders
            best_score = score
            continue
        if score < best_score:
            best_leaders = leaders
            best_score = score
        elif score == best_score and leaders < best_leaders:
            best_leaders = leaders

    if best_leaders is None:
        return [], [], float("inf"), False
    return best_leaders, induced_leader_edges(best_leaders, feasible_graph), best_score, True
