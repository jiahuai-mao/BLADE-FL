from __future__ import annotations

from itertools import product
from typing import Mapping, Set

from .types import ClientMetadata

Graph = Mapping[int, Set[int]]
MAX_EXACT_LEADER_COMBINATIONS = 50_000
LEADER_BEAM_WIDTH = 8_192


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

    if _combination_count_at_most(candidate_lists, MAX_EXACT_LEADER_COMBINATIONS):
        return _select_connected_leaders_exact(candidate_lists, clients_by_id, feasible_graph)
    return _select_connected_leaders_beam(candidate_lists, clients_by_id, feasible_graph)


def _select_connected_leaders_exact(
    candidate_lists: list[list[int]],
    clients_by_id: Mapping[int, ClientMetadata],
    feasible_graph: Graph,
) -> tuple[list[int], list[tuple[int, int]], float, bool]:
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


def _select_connected_leaders_beam(
    candidate_lists: list[list[int]],
    clients_by_id: Mapping[int, ClientMetadata],
    feasible_graph: Graph,
) -> tuple[list[int], list[tuple[int, int]], float, bool]:
    states: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
    for candidates in candidate_lists:
        next_states = []
        for score, leaders in states:
            for cid in candidates:
                next_states.append((score + clients_by_id[cid].train_time, leaders + (cid,)))
        next_states.sort(key=lambda item: (item[0], item[1]))
        states = next_states[:LEADER_BEAM_WIDTH]

    for score, leaders_tuple in states:
        leaders = list(leaders_tuple)
        if is_connected_subset(leaders, feasible_graph):
            return leaders, induced_leader_edges(leaders, feasible_graph), float(score), True
    return [], [], float("inf"), False


def _combination_count_at_most(candidate_lists: list[list[int]], limit: int) -> bool:
    count = 1
    for candidates in candidate_lists:
        count *= len(candidates)
        if count > limit:
            return False
    return True
