from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Dict, Set


Graph = Dict[int, Set[int]]


def build_client_graph(
    client_ids: Sequence[int],
    graph_type: str = "full",
    random_edge_prob: float = 0.3,
    seed: int = 0,
) -> Graph:
    nodes = sorted(int(cid) for cid in client_ids)
    if not nodes:
        raise ValueError("client_ids must not be empty.")

    graph_type = graph_type.lower()
    graph: Graph = {cid: set() for cid in nodes}
    if graph_type == "full":
        for src in nodes:
            graph[src] = {dst for dst in nodes if dst != src}
        return graph
    if graph_type == "ring":
        return _ring_graph(nodes)
    if graph_type == "random":
        graph = _ring_graph(nodes)
        rng = random.Random(seed)
        for left_idx, src in enumerate(nodes):
            for dst in nodes[left_idx + 1 :]:
                if dst in graph[src]:
                    continue
                if rng.random() < random_edge_prob:
                    graph[src].add(dst)
                    graph[dst].add(src)
        return graph
    raise ValueError(f"Unsupported graph_type: {graph_type}")


def graph_edges(graph: Mapping[int, set[int]]) -> list[tuple[int, int]]:
    edges = set()
    for src, neighbors in graph.items():
        for dst in neighbors:
            if src == dst:
                continue
            edges.add(tuple(sorted((int(src), int(dst)))))
    return sorted(edges)


def graph_from_edges(nodes: Sequence[int], edges: Sequence[tuple[int, int]]) -> Graph:
    graph: Graph = {int(node): set() for node in nodes}
    for left, right in edges:
        left = int(left)
        right = int(right)
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    return graph


def metropolis_weights(graph: Mapping[int, set[int]]) -> dict[int, dict[int, float]]:
    degrees = {node: len(neighbors) for node, neighbors in graph.items()}
    weights: dict[int, dict[int, float]] = {}
    for node, neighbors in graph.items():
        row: dict[int, float] = {}
        off_diag_sum = 0.0
        for neigh in sorted(neighbors):
            weight = 1.0 / (1.0 + max(degrees[node], degrees.get(neigh, 0)))
            row[neigh] = weight
            off_diag_sum += weight
        row[node] = 1.0 - off_diag_sum
        weights[node] = row
    return weights


def _ring_graph(nodes: list[int]) -> Graph:
    graph: Graph = {cid: set() for cid in nodes}
    if len(nodes) == 1:
        return graph
    for idx, src in enumerate(nodes):
        dst = nodes[(idx + 1) % len(nodes)]
        graph[src].add(dst)
        graph[dst].add(src)
    return graph
