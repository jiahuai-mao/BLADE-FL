from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clustered_dfl.clustering import (
    ClusteringParams,
    compute_model_transfer_weights,
    refresh_topology,
    select_best_k_topology,
    transfer_cluster_models,
)
from clustered_dfl.data import build_client_metadata, load_labels, partition_dirichlet, partition_iid
from clustered_dfl.models import create_model
from clustered_dfl.training import TrainingConfig
from clustered_dfl.training.data_loaders import build_client_states, load_torch_datasets, make_test_loader
from clustered_dfl.training.graph import build_client_graph, graph_from_edges, graph_edges, metropolis_weights
from clustered_dfl.training.local import train_client_from_vector
from clustered_dfl.training.metrics import evaluate_vector, model_divergence, weighted_global_model
from clustered_dfl.training.time import cluster_update_times
from clustered_dfl.training.vector_utils import average_vectors, initial_model_vector
from clustered_dfl.types import ClientMetadata, Topology
from scripts.run_training_simulation import (
    _create_unique_run_dir,
    _maybe_subset_test,
    _resolve_device,
    _resolve_k_max,
    _set_seed,
)


POLICIES = ("blade_refresh", "stale_topology", "full_reconstruction")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Figure 5 persistent leave/join dynamic-participation experiment.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "svhn"], default="cifar10")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-clients", type=int, default=50, help="Initial active client count.")
    parser.add_argument("--pool-clients", type=int, default=0, help="Total candidate client pool. Use 0 for num_clients plus reserve clients.")
    parser.add_argument("--reserve-ratio", type=float, default=0.6)
    parser.add_argument("--split", choices=["iid", "dirichlet"], default="dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--departure-ratio", type=float, default=0.4)
    parser.add_argument("--refresh-policy", choices=POLICIES, default="blade_refresh")
    parser.add_argument("--leave-event", type=int, default=700)
    parser.add_argument("--join-event", type=int, default=1400)
    parser.add_argument("--events", type=int, default=2100)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="resnet20", choices=["auto", "mlp", "cnnbn", "resnet20"])
    parser.add_argument("--k", default="auto")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", default="sqrt")
    parser.add_argument("--min-clients-per-cluster", type=int, default=2)
    parser.add_argument("--alpha-time", type=float, default=0.5)
    parser.add_argument("--alpha-distribution", type=float, default=0.5)
    parser.add_argument("--lambda-communication", type=float, default=0.2)
    parser.add_argument("--mixing-interval", type=int, default=1)
    parser.add_argument("--max-swaps", type=int, default=10)
    parser.add_argument("--baseline-graph", choices=["full", "ring", "random"], default="random")
    parser.add_argument("--random-edge-prob", type=float, default=0.45)
    parser.add_argument("--bytes-per-time-unit", type=float, default=1_000_000.0)
    parser.add_argument("--output-dir", default="results/training_figure5_dynamic_cifar10_a03")
    args = parser.parse_args()

    _validate_args(args)
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    pool_clients = _resolve_pool_clients(args)
    leave_count = max(1, int(round(args.num_clients * args.departure_ratio)))
    if pool_clients < args.num_clients + leave_count:
        raise ValueError("--pool-clients must provide enough reserve clients for the requested join event.")

    print(
        "[dynamic] start "
        f"dataset={args.dataset} policy={args.refresh_policy} ratio={args.departure_ratio:g} "
        f"seed={args.seed} active={args.num_clients} pool={pool_clients}",
        flush=True,
    )

    labels, num_classes = load_labels(args.dataset, args.data_dir)
    train_dataset, test_dataset, loaded_num_classes = load_torch_datasets(args.dataset, args.data_dir, seed=args.seed)
    if loaded_num_classes != num_classes:
        raise RuntimeError(f"label loader classes={num_classes}, dataset loader classes={loaded_num_classes}")
    test_dataset = _maybe_subset_test(test_dataset, args.test_samples, args.seed)
    test_loader = make_test_loader(test_dataset, args.batch_size, args.num_workers, device)

    client_indices = _partition_clients(args, labels, pool_clients)
    metadata = build_client_metadata(
        client_indices,
        labels,
        num_classes,
        time_mode="local_epochs_num_samples",
        local_epochs=args.local_epochs,
    )
    all_client_ids = [item.client_id for item in metadata]
    initial_active_ids = set(range(args.num_clients))
    reserve_ids = [cid for cid in all_client_ids if cid not in initial_active_ids]
    joined_ids = set(reserve_ids[:leave_count])

    graph_all = build_client_graph(all_client_ids, args.baseline_graph, args.random_edge_prob, args.seed)
    metadata_by_id = {item.client_id: item for item in metadata}
    active_metadata = [metadata_by_id[cid] for cid in sorted(initial_active_ids)]
    active_graph = _induced_graph(graph_all, initial_active_ids)
    topology, topology_metrics = _build_topology(args, active_metadata, active_graph)
    departed_ids = _select_departed_clients(topology, leave_count, args.seed)

    clients = build_client_states(
        train_dataset,
        client_indices,
        metadata,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
    )
    clients_by_id = {client.client_id: client for client in clients}
    model_factory = lambda: create_model(args.dataset, model_name=args.model)
    train_model = model_factory().to(device)
    eval_model = model_factory().to(device)
    config = TrainingConfig(
        algorithm="blade_dynamic",
        events=args.events,
        local_epochs=args.local_epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        eval_interval=args.eval_interval,
        device=device,
        mixing_interval=args.mixing_interval,
        seed=args.seed,
        lr_scheduler=args.lr_scheduler,
    )

    init = initial_model_vector(model_factory, args.seed, device)
    cluster_models = {idx: init.clone() for idx in range(len(topology.clusters))}
    counters = {idx: 0 for idx in cluster_models}
    current_time = 0.0
    transmitted = 0.0
    best_accuracy: float | None = None
    staleness_values: list[float] = []
    model_size_bytes = float(init.numel() * 4)
    active_ids = set(initial_active_ids)

    runtime = _make_runtime(topology, clients_by_id, cluster_models, args.local_epochs, start_time=current_time)
    metrics: list[dict[str, Any]] = []
    topology_events: list[dict[str, Any]] = [
        {
            "step": 0,
            "event_marker": "initial",
            "refresh_policy": args.refresh_policy,
            "refresh_mode": "initial",
            "active_client_count": len(active_ids),
            "K": len(topology.clusters),
            "departed_clients": "",
            "joined_clients": "",
            "leader_edges": str(topology.leader_edges),
        }
    ]

    best_accuracy = _append_metric(
        metrics,
        step=0,
        event_marker="initial",
        args=args,
        topology=topology,
        cluster_models=cluster_models,
        cluster_sample_counts=runtime["cluster_sample_counts"],
        model_factory=model_factory,
        eval_model=eval_model,
        test_loader=test_loader,
        device=device,
        current_time=current_time,
        train_loss=None,
        best_accuracy=best_accuracy,
        transmitted=transmitted,
        staleness_values=staleness_values,
        active_client_count=len(active_ids),
        pool_client_count=pool_clients,
        bytes_per_time_unit=args.bytes_per_time_unit,
    )
    last_eval_step = 0

    for event in range(1, args.events + 1):
        event_marker = "none"
        if event == args.leave_event:
            event_marker = "leave"
            active_ids -= departed_ids
            topology, cluster_models, counters, runtime, event_log = _apply_membership_event(
                args,
                event,
                event_marker,
                topology,
                cluster_models,
                counters,
                metadata_by_id,
                active_ids,
                graph_all,
                current_time,
                departed_ids=departed_ids,
                joined_ids=set(),
                clients_by_id=clients_by_id,
            )
            topology_events.append(event_log)
            best_accuracy = _append_metric_for_runtime(
                metrics,
                event,
                event_marker,
                args,
                topology,
                cluster_models,
                runtime,
                model_factory,
                eval_model,
                test_loader,
                device,
                current_time,
                None,
                best_accuracy,
                transmitted,
                staleness_values,
                len(active_ids),
                pool_clients,
            )
            last_eval_step = event
        elif event == args.join_event:
            event_marker = "join"
            active_ids |= joined_ids
            topology, cluster_models, counters, runtime, event_log = _apply_membership_event(
                args,
                event,
                event_marker,
                topology,
                cluster_models,
                counters,
                metadata_by_id,
                active_ids,
                graph_all,
                current_time,
                departed_ids=set(),
                joined_ids=joined_ids,
                clients_by_id=clients_by_id,
            )
            topology_events.append(event_log)
            best_accuracy = _append_metric_for_runtime(
                metrics,
                event,
                event_marker,
                args,
                topology,
                cluster_models,
                runtime,
                model_factory,
                eval_model,
                test_loader,
                device,
                current_time,
                None,
                best_accuracy,
                transmitted,
                staleness_values,
                len(active_ids),
                pool_clients,
            )
            last_eval_step = event

        if not runtime["heap"]:
            continue
        current_time, cluster_idx = _pop_next_available(runtime["heap"], set(cluster_models))
        if cluster_idx is None:
            continue
        start = cluster_models[cluster_idx]
        cluster_models[cluster_idx], train_loss = _update_cluster_model(
            topology,
            cluster_idx,
            start,
            clients_by_id,
            model_factory,
            train_model,
            config,
        )
        counters[cluster_idx] = counters.get(cluster_idx, 0) + 1
        transmitted += 2.0 * model_size_bytes * max(len(topology.clusters[cluster_idx]) - 1, 0)

        if args.mixing_interval > 0 and counters[cluster_idx] % args.mixing_interval == 0:
            row = runtime["weights"][cluster_idx]
            vectors = []
            weights = []
            for node, weight in row.items():
                weights.append(weight)
                if node == cluster_idx:
                    vectors.append(cluster_models[cluster_idx])
                else:
                    vectors.append(runtime["cache"][cluster_idx][node])
                    stale = counters.get(node, 0) - runtime["cache_counters"][cluster_idx].get(node, 0)
                    staleness_values.append(float(stale))
            cluster_models[cluster_idx] = average_vectors(vectors, weights)
            transmitted += 2.0 * model_size_bytes * len(runtime["cluster_graph"][cluster_idx])

        for neigh in runtime["cluster_graph"][cluster_idx]:
            runtime["cache"][neigh][cluster_idx] = cluster_models[cluster_idx].clone()
            runtime["cache_counters"][neigh][cluster_idx] = counters[cluster_idx]
        runtime["push"](current_time + runtime["cluster_times"][cluster_idx], cluster_idx)

        if (event % args.eval_interval == 0 or event == args.events) and event != last_eval_step:
            best_accuracy = _append_metric_for_runtime(
                metrics,
                event,
                event_marker,
                args,
                topology,
                cluster_models,
                runtime,
                model_factory,
                eval_model,
                test_loader,
                device,
                current_time,
                train_loss,
                best_accuracy,
                transmitted,
                staleness_values,
                len(active_ids),
                pool_clients,
            )
            last_eval_step = event

    event_summary = _compute_event_summary(metrics, args.leave_event, args.join_event)
    final_summary = {
        **metrics[-1],
        **event_summary,
        "departed_clients": sorted(departed_ids),
        "joined_clients": sorted(joined_ids),
        "initial_topology_metrics": topology_metrics,
    }

    run_name = _run_name(args, pool_clients)
    run_dir = _create_unique_run_dir(Path(args.output_dir), run_name)
    _write_json(run_dir / "config.json", {**vars(args), "pool_clients_resolved": pool_clients, "run_dir": str(run_dir)})
    _write_json(run_dir / "clients.json", [_metadata_to_dict(item) for item in metadata])
    _write_json(run_dir / "initial_topology.json", _topology_to_dict(topology_events[0], topology_metrics))
    _write_rows(run_dir / "metrics.csv", metrics)
    _write_rows(run_dir / "topology_events.csv", topology_events)
    _write_rows(run_dir / "event_summary.csv", [event_summary])
    _write_json(run_dir / "final_summary.json", final_summary)

    print(
        "[dynamic] finished "
        f"policy={args.refresh_policy} ratio={args.departure_ratio:g} seed={args.seed} "
        f"acc={float(metrics[-1].get('test_accuracy', 0.0)):.4f} output_dir={run_dir}",
        flush=True,
    )


def _validate_args(args) -> None:
    if not (0.0 < args.departure_ratio < 1.0):
        raise ValueError("--departure-ratio must be in (0, 1).")
    if not (0 < args.leave_event < args.join_event < args.events):
        raise ValueError("Require 0 < --leave-event < --join-event < --events.")
    if args.bytes_per_time_unit <= 0.0:
        raise ValueError("--bytes-per-time-unit must be positive.")


def _resolve_pool_clients(args) -> int:
    if args.pool_clients > 0:
        return args.pool_clients
    reserve = int(math.ceil(args.num_clients * args.reserve_ratio))
    leave_count = int(round(args.num_clients * args.departure_ratio))
    return args.num_clients + max(reserve, leave_count)


def _partition_clients(args, labels: np.ndarray, pool_clients: int) -> list[list[int]]:
    if args.split == "iid":
        return partition_iid(labels, pool_clients, 1.0, args.seed)
    return partition_dirichlet(labels, pool_clients, args.dirichlet_alpha, 1.0, args.seed)


def _build_topology(args, metadata: list[ClientMetadata], graph) -> tuple[Topology, dict]:
    k_max = _resolve_k_max(args.k_max, len(metadata))
    if args.k.lower() == "auto":
        return select_best_k_topology(
            metadata,
            k_min=args.k_min,
            k_max=k_max,
            min_clients_per_cluster=args.min_clients_per_cluster,
            alpha_time=args.alpha_time,
            alpha_distribution=args.alpha_distribution,
            lambda_communication=args.lambda_communication,
            mixing_interval=args.mixing_interval,
            max_swaps=args.max_swaps,
            feasible_graph=graph,
        )
    return select_best_k_topology(
        metadata,
        k_min=int(args.k),
        k_max=int(args.k),
        min_clients_per_cluster=args.min_clients_per_cluster,
        alpha_time=args.alpha_time,
        alpha_distribution=args.alpha_distribution,
        lambda_communication=args.lambda_communication,
        mixing_interval=args.mixing_interval,
        max_swaps=args.max_swaps,
        feasible_graph=graph,
    )


def _select_departed_clients(topology: Topology, leave_count: int, seed: int) -> set[int]:
    rng = random.Random(seed + 7919)
    selected: list[int] = []
    quotas = [int(math.floor(leave_count * len(cluster) / sum(len(c) for c in topology.clusters))) for cluster in topology.clusters]
    while sum(quotas) < leave_count:
        idx = max(range(len(topology.clusters)), key=lambda item: len(topology.clusters[item]) - quotas[item])
        quotas[idx] += 1
    for cluster_idx, cluster in enumerate(topology.clusters):
        candidates = list(cluster)
        rng.shuffle(candidates)
        selected.extend(candidates[: min(quotas[cluster_idx], max(len(cluster) - 1, 0))])
    remaining = [cid for cluster in topology.clusters for cid in cluster if cid not in selected]
    rng.shuffle(remaining)
    while len(selected) < leave_count and remaining:
        selected.append(remaining.pop())
    return set(selected[:leave_count])


def _apply_membership_event(
    args,
    step: int,
    event_marker: str,
    topology: Topology,
    cluster_models: dict[int, torch.Tensor],
    counters: dict[int, int],
    metadata_by_id: dict[int, ClientMetadata],
    active_ids: set[int],
    graph_all,
    current_time: float,
    departed_ids: set[int],
    joined_ids: set[int],
    clients_by_id,
):
    previous_topology = topology
    previous_models = [cluster_models[idx] for idx in range(len(previous_topology.clusters))]
    previous_counters = dict(counters)
    active_metadata = [metadata_by_id[cid] for cid in sorted(active_ids)]
    active_graph = _induced_graph(graph_all, active_ids)
    params = _clustering_params(args, len(active_metadata))

    if args.refresh_policy == "blade_refresh":
        new_topology, log = refresh_topology(
            previous_topology=previous_topology,
            previous_clients=[metadata_by_id[cid] for cid in sorted(_topology_client_ids(previous_topology))],
            updated_clients=active_metadata,
            previous_cluster_models=previous_models,
            updated_feasible_graph=active_graph,
            params=params,
            refresh_round=step,
            departed_clients=departed_ids,
            newly_joined_clients=joined_ids,
        )
        if new_topology is None:
            new_topology = _stale_topology(previous_topology, active_ids, metadata_by_id, active_graph)
            transfer_weights = compute_model_transfer_weights(previous_topology, new_topology, metadata_by_id)
            log["refresh_mode"] = "stale_fallback"
        else:
            transfer_weights = log.get("model_transfer_weights") or compute_model_transfer_weights(previous_topology, new_topology, metadata_by_id)
    elif args.refresh_policy == "full_reconstruction":
        new_topology, metrics = select_best_k_topology(
            active_metadata,
            k_min=params.k_min,
            k_max=params.k_max,
            min_clients_per_cluster=params.min_clients_per_cluster,
            alpha_time=params.alpha_time,
            alpha_distribution=params.alpha_distribution,
            lambda_communication=params.lambda_communication,
            mixing_interval=params.mixing_interval,
            max_swaps=params.max_swaps,
            feasible_graph=active_graph,
        )
        transfer_weights = compute_model_transfer_weights(previous_topology, new_topology, metadata_by_id)
        log = {"refresh_mode": "reconstruction", "repair_attempted": False, "repair_succeeded": False, "reconstruction_used": True, "metrics": metrics}
    else:
        new_topology = _stale_topology(previous_topology, active_ids, metadata_by_id, active_graph)
        transfer_weights = compute_model_transfer_weights(previous_topology, new_topology, metadata_by_id)
        log = {"refresh_mode": "stale", "repair_attempted": False, "repair_succeeded": False, "reconstruction_used": False}

    transferred_models = transfer_cluster_models(previous_models, transfer_weights)
    new_cluster_models = {
        idx: model.clone() if model is not None else previous_models[0].clone()
        for idx, model in enumerate(transferred_models)
    }
    new_counters = _transfer_counters(previous_topology, new_topology, previous_counters, metadata_by_id)
    runtime = _make_runtime(new_topology, clients_by_id, new_cluster_models, args.local_epochs, start_time=current_time)
    event_log = {
        "step": step,
        "event_marker": event_marker,
        "refresh_policy": args.refresh_policy,
        "refresh_mode": log.get("refresh_mode", ""),
        "repair_attempted": bool(log.get("repair_attempted", False)),
        "repair_succeeded": bool(log.get("repair_succeeded", False)),
        "reconstruction_used": bool(log.get("reconstruction_used", False)),
        "active_client_count": len(active_ids),
        "topology_client_count": sum(len(cluster) for cluster in new_topology.clusters),
        "K": len(new_topology.clusters),
        "departed_clients": " ".join(str(cid) for cid in sorted(departed_ids)),
        "joined_clients": " ".join(str(cid) for cid in sorted(joined_ids)),
        "leader_edges": str(new_topology.leader_edges),
    }
    return new_topology, new_cluster_models, new_counters, runtime, event_log


def _clustering_params(args, active_count: int) -> ClusteringParams:
    return ClusteringParams(
        k_min=args.k_min,
        k_max=_resolve_k_max(args.k_max, active_count) if args.k.lower() == "auto" else int(args.k),
        min_clients_per_cluster=args.min_clients_per_cluster,
        alpha_time=args.alpha_time,
        alpha_distribution=args.alpha_distribution,
        lambda_communication=args.lambda_communication,
        mixing_interval=args.mixing_interval,
        max_swaps=args.max_swaps,
    )


def _stale_topology(
    previous_topology: Topology,
    active_ids: set[int],
    metadata_by_id: dict[int, ClientMetadata],
    active_graph,
) -> Topology:
    clusters = [sorted(cid for cid in cluster if cid in active_ids) for cluster in previous_topology.clusters]
    kept = [(old_idx, cluster) for old_idx, cluster in enumerate(clusters) if cluster]
    if not kept:
        raise RuntimeError("stale topology has no active clusters.")
    old_to_new = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(kept)}
    new_clusters = [cluster for _, cluster in kept]
    leaders = []
    for old_idx, cluster in kept:
        old_leader = previous_topology.leaders[old_idx]
        if old_leader in cluster:
            leaders.append(old_leader)
        else:
            leaders.append(min(cluster, key=lambda cid: metadata_by_id[cid].train_time))
    edges = []
    for left_old, right_old in _cluster_index_edges(previous_topology):
        if left_old in old_to_new and right_old in old_to_new:
            left = leaders[old_to_new[left_old]]
            right = leaders[old_to_new[right_old]]
            if right in active_graph.get(left, set()):
                edges.append(tuple(sorted((left, right))))
    return Topology(
        clusters=new_clusters,
        leaders=leaders,
        leader_edges=sorted(set(edges)),
        client_to_cluster={cid: idx for idx, cluster in enumerate(new_clusters) for cid in cluster},
    )


def _cluster_index_edges(topology: Topology) -> list[tuple[int, int]]:
    leader_to_cluster = {leader: idx for idx, leader in enumerate(topology.leaders)}
    edges = []
    for left, right in topology.leader_edges:
        if left in leader_to_cluster and right in leader_to_cluster:
            edges.append(tuple(sorted((leader_to_cluster[left], leader_to_cluster[right]))))
    return sorted(set(edges))


def _transfer_counters(
    previous_topology: Topology,
    new_topology: Topology,
    previous_counters: dict[int, int],
    metadata_by_id: dict[int, ClientMetadata],
) -> dict[int, int]:
    weights = compute_model_transfer_weights(
        previous_topology,
        new_topology,
        metadata_by_id,
    )
    counters = {}
    for new_idx, row in weights.items():
        if not row:
            counters[int(new_idx)] = 0
        else:
            counters[int(new_idx)] = int(round(sum(previous_counters.get(int(old_idx), 0) * float(weight) for old_idx, weight in row.items())))
    return counters


def _make_runtime(
    topology: Topology,
    clients_by_id,
    cluster_models: dict[int, torch.Tensor],
    local_epochs: int,
    start_time: float = 0.0,
):
    import heapq

    cluster_graph = _cluster_graph(topology)
    weights = metropolis_weights(cluster_graph)
    cluster_times = cluster_update_times(clients_by_id, topology, local_epochs)
    cache = {
        cluster_idx: {neigh: cluster_models[neigh].clone() for neigh in cluster_graph[cluster_idx]}
        for cluster_idx in cluster_models
    }
    cache_counters = {
        cluster_idx: {neigh: 0 for neigh in cluster_graph[cluster_idx]}
        for cluster_idx in cluster_models
    }
    heap = [(float(start_time) + cluster_times[idx], idx) for idx in cluster_models]
    heapq.heapify(heap)

    def push(item_time: float, cluster_idx: int) -> None:
        heapq.heappush(heap, (item_time, cluster_idx))

    return {
        "cluster_graph": cluster_graph,
        "weights": weights,
        "cluster_times": cluster_times,
        "cluster_sample_counts": [
            sum(clients_by_id[cid].metadata.num_samples for cid in cluster)
            for cluster in topology.clusters
        ],
        "cache": cache,
        "cache_counters": cache_counters,
        "heap": heap,
        "push": push,
    }


def _cluster_graph(topology: Topology):
    leader_to_cluster = {leader: idx for idx, leader in enumerate(topology.leaders)}
    edges = []
    for left, right in topology.leader_edges:
        if left in leader_to_cluster and right in leader_to_cluster:
            edges.append((leader_to_cluster[left], leader_to_cluster[right]))
    return graph_from_edges(list(range(len(topology.clusters))), edges)


def _pop_next_available(heap, active_cluster_ids: set[int]):
    import heapq

    while heap:
        current_time, cluster_idx = heapq.heappop(heap)
        if cluster_idx in active_cluster_ids:
            return current_time, cluster_idx
    return 0.0, None


def _update_cluster_model(topology, cluster_idx, start, clients_by_id, model_factory, train_model, config):
    local_vectors = []
    local_weights = []
    losses = []
    for cid in topology.clusters[cluster_idx]:
        client = clients_by_id[cid]
        vector, loss = train_client_from_vector(model_factory, start, client, config, model=train_model)
        local_vectors.append(vector)
        local_weights.append(float(client.metadata.num_samples))
        losses.append(loss)
    train_loss = sum(losses) / len(losses) if losses else None
    return average_vectors(local_vectors, local_weights), train_loss


def _append_metric_for_runtime(
    metrics,
    step,
    event_marker,
    args,
    topology,
    cluster_models,
    runtime,
    model_factory,
    eval_model,
    test_loader,
    device,
    current_time,
    train_loss,
    best_accuracy,
    transmitted,
    staleness_values,
    active_client_count,
    pool_client_count,
):
    return _append_metric(
        metrics,
        step,
        event_marker,
        args,
        topology,
        cluster_models,
        runtime["cluster_sample_counts"],
        model_factory,
        eval_model,
        test_loader,
        device,
        current_time,
        train_loss,
        best_accuracy,
        transmitted,
        staleness_values,
        active_client_count,
        pool_client_count,
        args.bytes_per_time_unit,
    )


def _append_metric(
    rows,
    step,
    event_marker,
    args,
    topology,
    cluster_models,
    cluster_sample_counts,
    model_factory,
    eval_model,
    test_loader,
    device,
    current_time,
    train_loss,
    best_accuracy,
    transmitted,
    staleness_values,
    active_client_count,
    pool_client_count,
    bytes_per_time_unit,
):
    global_model = weighted_global_model(
        [cluster_models[idx] for idx in range(len(topology.clusters))],
        cluster_sample_counts,
    )
    test_loss, test_accuracy, test_macro_f1 = evaluate_vector(model_factory, global_model, test_loader, device, model=eval_model)
    if test_accuracy is not None:
        best_accuracy = test_accuracy if best_accuracy is None else max(best_accuracy, test_accuracy)
    mean_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0.0
    max_staleness = max(staleness_values) if staleness_values else 0.0
    communication_time = transmitted / bytes_per_time_unit
    row = {
        "step": int(step),
        "algorithm": "clustered_async_dynamic",
        "refresh_policy": args.refresh_policy,
        "departure_ratio": float(args.departure_ratio),
        "seed": int(args.seed),
        "event_marker": event_marker,
        "leave_event": int(args.leave_event),
        "join_event": int(args.join_event),
        "K": len(topology.clusters),
        "active_client_count": int(active_client_count),
        "population_client_count": int(active_client_count),
        "topology_client_count": int(sum(len(cluster) for cluster in topology.clusters)),
        "pool_client_count": int(pool_client_count),
        "virtual_time": float(current_time),
        "communication_time_proxy": float(communication_time),
        "wall_clock_proxy": float(current_time + communication_time),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "best_accuracy": best_accuracy,
        "mean_staleness": float(mean_staleness),
        "max_staleness": float(max_staleness),
        "leader_edges": str(topology.leader_edges),
        "transmitted_bytes_proxy": float(transmitted),
        "model_divergence": model_divergence(list(cluster_models.values()), global_model),
    }
    rows.append(row)
    if step > 0:
        print(
            "[progress] "
            f"policy={args.refresh_policy} ratio={args.departure_ratio:g} "
            f"[{step}/{args.events}] acc={float(test_accuracy or 0.0):.4f} "
            f"best={float(best_accuracy or 0.0):.4f} marker={event_marker}",
            flush=True,
        )
    return best_accuracy


def _compute_event_summary(metrics: list[dict[str, Any]], leave_event: int, join_event: int) -> dict[str, Any]:
    rows = [row for row in metrics if row.get("test_accuracy") is not None]
    pre_rows = [row for row in rows if int(row["step"]) < leave_event]
    after_leave = [row for row in rows if leave_event <= int(row["step"]) <= join_event]
    pre_leave_accuracy = float(pre_rows[-1]["test_accuracy"]) if pre_rows else None
    if pre_leave_accuracy is None or not after_leave:
        drop = None
        min_acc = None
        recovery_event = None
        recovery_time = None
    else:
        min_row = min(after_leave, key=lambda row: float(row["test_accuracy"]))
        min_acc = float(min_row["test_accuracy"])
        drop = max(pre_leave_accuracy - min_acc, 0.0)
        recovered = [row for row in after_leave if float(row["test_accuracy"]) >= pre_leave_accuracy]
        recovery_event = int(recovered[0]["step"]) if recovered else None
        recovery_time = (recovery_event - leave_event) if recovery_event is not None else None
    final_accuracy = float(rows[-1]["test_accuracy"]) if rows else None
    return {
        "pre_leave_accuracy": pre_leave_accuracy,
        "min_accuracy_after_leave": min_acc,
        "accuracy_drop_after_leave": drop,
        "recovery_event_after_leave": recovery_event,
        "recovery_time_after_leave": recovery_time,
        "final_accuracy": final_accuracy,
        "leave_event": leave_event,
        "join_event": join_event,
    }


def _induced_graph(graph, active_ids: set[int]):
    return {cid: {neigh for neigh in graph.get(cid, set()) if neigh in active_ids} for cid in sorted(active_ids)}


def _topology_client_ids(topology: Topology) -> set[int]:
    return {cid for cluster in topology.clusters for cid in cluster}


def _metadata_to_dict(item: ClientMetadata) -> dict[str, Any]:
    return asdict(item)


def _topology_to_dict(event_row: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {"event": event_row, "metrics": metrics}


def _run_name(args, pool_clients: int) -> str:
    return (
        f"{args.dataset}_dirichlet_a{args.dirichlet_alpha:g}_"
        f"c{args.num_clients}_pool{pool_clients}_"
        f"{args.refresh_policy}_leave{args.departure_ratio:g}_"
        f"event{args.events}_l{args.leave_event}_j{args.join_event}_"
        f"mix{args.mixing_interval}_seed{args.seed}"
    )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return "<tensor>"
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(value) for value in obj]
    return obj


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    main()
