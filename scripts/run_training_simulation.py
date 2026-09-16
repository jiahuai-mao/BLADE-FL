from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clustered_dfl.clustering import build_deterministic_balanced_clusters, select_best_k_topology
from clustered_dfl.clustering import compute_capacities
from clustered_dfl.data import build_client_metadata, load_labels, partition_dirichlet, partition_iid
from clustered_dfl.models import create_model
from clustered_dfl.topology import select_connected_leaders
from clustered_dfl.training import (
    ADPSGDRunner,
    ClusteredAsyncClipProxTrainingRunner,
    ClusteredAsyncTrainingRunner,
    ClusteredAsyncV1TrainingRunner,
    ClusteredBarrierTrainingRunner,
    ClusteredTrainingRunner,
    DPSGDRunner,
    FedAvgRunner,
    MDFeelTrainingRunner,
    TrainingConfig,
)
from clustered_dfl.training.mdfeel import build_mdfeel_topology, select_best_mdfeel_topology
from clustered_dfl.training.data_loaders import build_client_states, load_torch_datasets, make_test_loader
from clustered_dfl.training.graph import build_client_graph, graph_edges
from clustered_dfl.types import Topology


METRIC_FIELDS = [
    "step",
    "algorithm",
    "K",
    "virtual_time",
    "train_loss",
    "test_loss",
    "test_accuracy",
    "test_macro_f1",
    "best_accuracy",
    "mean_staleness",
    "max_staleness",
    "leader_edges",
    "transmitted_bytes_proxy",
    "model_divergence",
    "sync_comm_time_proxy",
    "step_sync_comm_time_proxy",
    "sync_comm_bandwidth_proxy",
]

GRAPHED_ALGORITHMS = {
    "clustered",
    "clustered_barrier",
    "clustered_no_mixing",
    "clustered_async",
    "clustered_asyncv1",
    "clustered_async_clipprox",
    "clustered_async_no_mixing",
    "random_clustered_async",
    "dpsgd",
    "adpsgd",
    "mdfeel",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run client training simulation for clustered, D-PSGD, and AD-PSGD.")
    parser.add_argument(
        "--algorithm",
        choices=[
            "clustered",
            "clustered_barrier",
            "clustered_no_mixing",
            "clustered_async",
            "clustered_asyncv1",
            "clustered_async_clipprox",
            "clustered_async_no_mixing",
            "random_clustered_async",
            "fedavg",
            "dpsgd",
            "adpsgd",
            "mdfeel",
            "core3",
            "all",
        ],
        default="clustered",
    )
    parser.add_argument("--dataset", choices=["synthetic", "mnist", "fashionmnist", "cifar10", "cifar100", "svhn", "hhar"], default="synthetic")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--split", choices=["iid", "dirichlet"], default="dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--samples-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0, help="Use 0 for the full test set.")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--model", default="auto", choices=["auto", "mlp", "cnnbn", "resnet20", "hhar1dcnn"], help="Use auto to select CnnBN for CIFAR-like datasets and MLP otherwise.")
    parser.add_argument(
        "--hhar-feature-mode",
        default="auto",
        choices=["auto", "stats", "raw"],
        help="HHAR only: stats uses 12 window statistics for MLP; raw uses 3x128 windows for hhar1dcnn.",
    )
    parser.add_argument("--hhar-window-size", type=int, default=128)
    parser.add_argument("--hhar-step-size", type=int, default=64)
    parser.add_argument("--hhar-sensor-mode", choices=["auto", "acc", "acc_gyro"], default="auto")
    parser.add_argument("--hhar-test-split", choices=["window", "group"], default="window")
    parser.add_argument("--hhar-input-channels", type=int, default=0, help="HHAR raw model channels. Use 0 to infer from loaded tensors.")
    parser.add_argument(
        "--hhar-client-partition",
        choices=["dirichlet", "group"],
        default="dirichlet",
        help="HHAR only: group packs user-model-device groups into --num-clients clients.",
    )
    parser.add_argument("--k", default="auto", help="Clustered algorithms only: fixed integer K or `auto`.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", default="sqrt", help="Clustered algorithms only: integer K upper bound, `sqrt`, or `none`.")
    parser.add_argument("--min-clients-per-cluster", type=int, default=2)
    parser.add_argument("--alpha-time", type=float, default=0.5)
    parser.add_argument("--alpha-distribution", type=float, default=0.5)
    parser.add_argument("--lambda-communication", type=float, default=0.2)
    parser.add_argument("--mixing-interval", type=int, default=5)
    parser.add_argument(
        "--sync-comm-bandwidth-proxy",
        type=float,
        default=1_000_000.0,
        help="clustered_barrier only: bytes per time-proxy unit for blocking synchronous leader communication.",
    )
    parser.add_argument("--clipprox-mu", type=float, default=0.0, help="clustered_async_clipprox only: FedProx proximal strength.")
    parser.add_argument("--clipprox-clip-norm", type=float, default=1.0, help="clustered_async_clipprox only: L2 delta clipping norm C.")
    parser.add_argument(
        "--cluster-delta-clip-norm",
        type=float,
        default=0.0,
        help="clustered_asyncv1 only: L2 norm clip applied to the corrected cluster delta; use 0 to disable.",
    )
    parser.add_argument(
        "--max-async-update-skew",
        type=int,
        default=1,
        help="AD-PSGD fairness bound: a client can be at most this many updates ahead of the slowest client; use -1 to disable.",
    )
    parser.add_argument("--max-swaps", type=int, default=10)
    parser.add_argument("--baseline-graph", choices=["full", "ring", "random"], default="random")
    parser.add_argument("--random-edge-prob", type=float, default=0.3)
    parser.add_argument("--random-topology-attempts", type=int, default=100)
    parser.add_argument("--output-dir", default="results/training")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Replace the requested algorithm result directory instead of skipping a complete run.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    hhar_feature_mode = _resolve_hhar_feature_mode(args.dataset, args.model, args.hhar_feature_mode)
    print(
        f"[training] start dataset={args.dataset} algorithm={args.algorithm} "
        f"num_clients={args.num_clients} seed={args.seed}",
        flush=True,
    )
    if args.dataset.lower() == "hhar":
        print("[training] loading HHAR dataset once for labels and tensors...", flush=True)
        train_dataset, test_dataset, loaded_num_classes = load_torch_datasets(
            args.dataset,
            args.data_dir,
            seed=args.seed,
            hhar_feature_mode=hhar_feature_mode,
            hhar_window_size=args.hhar_window_size,
            hhar_step_size=args.hhar_step_size,
            hhar_sensor_mode=args.hhar_sensor_mode,
            hhar_test_split=args.hhar_test_split,
        )
        labels = _labels_from_tensor_dataset(train_dataset)
        num_classes = loaded_num_classes
    else:
        print("[training] loading labels...", flush=True)
        labels, num_classes = load_labels(args.dataset, args.data_dir)
        print("[training] loading dataset tensors...", flush=True)
        train_dataset, test_dataset, loaded_num_classes = load_torch_datasets(args.dataset, args.data_dir, seed=args.seed)
    if loaded_num_classes != num_classes:
        raise RuntimeError(f"label loader classes={num_classes}, dataset loader classes={loaded_num_classes}")
    test_dataset = _maybe_subset_test(test_dataset, args.test_samples, args.seed)
    test_loader = make_test_loader(test_dataset, args.batch_size, args.num_workers, device)

    print(f"[training] partitioning {len(labels)} samples with split={args.split}...", flush=True)
    client_indices = _partition_clients(args, labels, train_dataset)
    metadata = build_client_metadata(
        client_indices,
        labels,
        num_classes,
        time_mode="local_epochs_num_samples",
        local_epochs=args.local_epochs,
    )
    client_ids = [item.client_id for item in metadata]
    topology = None
    cluster_metrics = None
    mdfeel_topology = None
    mdfeel_cluster_metrics = None
    random_topology = None
    random_cluster_metrics = None
    feasible_graph = None
    needs_balanced_topology = args.algorithm in {"clustered", "clustered_barrier", "clustered_no_mixing", "clustered_async", "clustered_asyncv1", "clustered_async_clipprox", "clustered_async_no_mixing", "core3", "all"}
    needs_mdfeel_topology = args.algorithm in {"mdfeel"}
    needs_random_topology = args.algorithm in {"random_clustered_async", "all"}
    needs_feasible_graph = args.algorithm in {"core3", "all"} or args.algorithm in GRAPHED_ALGORITHMS
    if needs_feasible_graph:
        print(
            f"[training] building shared initial feasible graph "
            f"mode={args.baseline_graph}...",
            flush=True,
        )
        feasible_graph = build_client_graph(client_ids, args.baseline_graph, args.random_edge_prob, args.seed)

    if needs_balanced_topology:
        print(f"[training] building balanced topology k={args.k}...", flush=True)
        topology_start = time.perf_counter()
        topology, cluster_metrics = _build_topology(args, metadata, feasible_graph)
        topology_wall_time = time.perf_counter() - topology_start
        cluster_metrics["construction_wall_time_sec"] = float(topology_wall_time)
        print(
            f"[training] balanced topology ready K={len(topology.clusters)} "
            f"construction_wall_time_sec={topology_wall_time:.6f}.",
            flush=True,
        )
    if needs_mdfeel_topology:
        print(f"[training] building adapted MD-FEEL topology k={args.k}...", flush=True)
        topology_start = time.perf_counter()
        mdfeel_topology, mdfeel_cluster_metrics = _build_mdfeel_topology(args, metadata, feasible_graph)
        topology_wall_time = time.perf_counter() - topology_start
        mdfeel_cluster_metrics["construction_wall_time_sec"] = float(topology_wall_time)
        print(
            f"[training] adapted MD-FEEL topology ready K={len(mdfeel_topology.clusters)} "
            f"construction_wall_time_sec={topology_wall_time:.6f}.",
            flush=True,
        )
    if needs_random_topology:
        active_k = len(topology.clusters) if topology is not None else _resolve_fixed_or_auto_k(args, metadata, feasible_graph)
        print(f"[training] building random topology K={active_k}...", flush=True)
        try:
            random_topology, random_cluster_metrics = _build_random_topology(
                metadata,
                active_k,
                args.seed,
                feasible_graph,
                args.random_topology_attempts,
            )
        except RuntimeError as exc:
            if args.algorithm != "all":
                raise
            random_cluster_metrics = {
                "K": active_k,
                "connected": False,
                "method": "random_balanced",
                "attempts": args.random_topology_attempts,
                "error": str(exc),
            }
            print(f"[training] warning: skipping random_clustered_async: {exc}", flush=True)

    hhar_input_channels = _resolve_hhar_input_channels(args, train_dataset)
    model_factory = lambda: create_model(
        args.dataset,
        hidden_dim=args.hidden_dim,
        model_name=args.model,
        hhar_input_channels=hhar_input_channels,
    )
    run_algorithms = (
        ["clustered_async", "clustered_async_clipprox", "fedavg", "dpsgd", "adpsgd", "clustered_async_no_mixing", "random_clustered_async"]
        if args.algorithm == "all"
        else [args.algorithm]
    )
    if args.algorithm == "core3":
        run_algorithms = ["clustered_async", "dpsgd", "adpsgd"]
    if random_topology is None:
        run_algorithms = [algorithm for algorithm in run_algorithms if algorithm != "random_clustered_async"]
    requested_run_name = args.run_name or _default_run_name(args, topology or mdfeel_topology)
    run_dir = _prepare_run_dir(Path(args.output_dir), requested_run_name)
    shared_dir = run_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    if args.algorithm == "clustered_asyncv1" and (shared_dir / "config.json").exists():
        print(f"[training] reusing existing shared artifacts for clustered_asyncv1: {shared_dir}", flush=True)
    else:
        _save_shared_artifacts(
            shared_dir,
            args,
            hhar_feature_mode,
            device,
            requested_run_name,
            run_dir,
            metadata,
            topology,
            random_topology,
            cluster_metrics,
            mdfeel_topology,
            mdfeel_cluster_metrics,
            random_cluster_metrics,
            feasible_graph,
            run_algorithms,
        )

    all_metrics: list[dict] = []
    summaries: dict[str, dict] = {}
    for algorithm in run_algorithms:
        active_mixing_interval = 0 if algorithm in {"clustered_no_mixing", "clustered_async_no_mixing"} else args.mixing_interval
        algorithm_dir = run_dir / _algorithm_run_name(algorithm, args, active_mixing_interval)
        if args.force_rerun and algorithm_dir.exists():
            print(f"[training] replacing existing algorithm={algorithm} output_dir={algorithm_dir}", flush=True)
            shutil.rmtree(algorithm_dir)
        if _is_algorithm_complete(algorithm_dir):
            print(f"[training] skip existing algorithm={algorithm} output_dir={algorithm_dir}", flush=True)
            summaries[algorithm] = _load_json(algorithm_dir / "final_summary.json")
            continue

        print(f"[training] running {algorithm}...", flush=True)
        _set_seed(args.seed)
        clients = build_client_states(
            train_dataset,
            client_indices,
            metadata,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            device=device,
        )
        config = TrainingConfig(
            algorithm=algorithm,
            rounds=args.rounds,
            events=args.events,
            local_epochs=args.local_epochs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eval_interval=args.eval_interval,
            device=device,
            mixing_interval=active_mixing_interval,
            seed=args.seed,
            max_async_update_skew=args.max_async_update_skew,
            clipprox_mu=args.clipprox_mu,
            clipprox_clip_norm=args.clipprox_clip_norm,
            cluster_delta_clip_norm=args.cluster_delta_clip_norm,
            lr_scheduler=args.lr_scheduler,
            sync_comm_bandwidth_proxy=args.sync_comm_bandwidth_proxy,
        )
        runner = _make_runner(
            algorithm,
            clients,
            topology,
            mdfeel_topology,
            random_topology,
            feasible_graph,
            model_factory,
            test_loader,
            config,
        )
        train_start = time.perf_counter()
        artifacts = runner.run()
        training_wall_time = time.perf_counter() - train_start
        artifacts.summary["training_wall_time_sec"] = float(training_wall_time)
        active_topology_metrics = mdfeel_cluster_metrics if algorithm == "mdfeel" else cluster_metrics
        if active_topology_metrics is not None and (algorithm.startswith("clustered") or algorithm == "mdfeel"):
            artifacts.summary["topology_construction_wall_time_sec"] = float(
                active_topology_metrics.get("construction_wall_time_sec", 0.0)
            )
            if training_wall_time > 0:
                artifacts.summary["topology_overhead_relative_to_training_wall_time"] = float(
                    active_topology_metrics.get("construction_wall_time_sec", 0.0) / training_wall_time
                )
        all_metrics.extend(artifacts.metrics)
        summaries[algorithm] = artifacts.summary
        algorithm_dir.mkdir(parents=True, exist_ok=True)
        _save_json(
            algorithm_dir / "config.json",
            {
                **vars(args),
                "algorithm": algorithm,
                "device": device,
                "active_mixing_interval": active_mixing_interval,
                "shared_dir": str(shared_dir),
                "result_dir": str(algorithm_dir),
            },
        )
        _write_metrics_csv(algorithm_dir / "metrics.csv", artifacts.metrics)
        for table_name, rows in artifacts.extra_tables.items():
            _write_rows_csv(algorithm_dir / f"{table_name}.csv", rows)
        _save_json(algorithm_dir / "final_summary.json", artifacts.summary)
        print(f"[training] finished {algorithm}.", flush=True)

    if all_metrics:
        _write_metrics_csv(run_dir / "metrics_combined.csv", all_metrics)
    _save_json(run_dir / "final_summary.json", summaries if args.algorithm in {"core3", "all"} else summaries[run_algorithms[0]])

    print(f"[training] output_dir={run_dir}")
    for algorithm, summary in summaries.items():
        print(
            "[training] "
            f"algorithm={algorithm} "
            f"step={summary.get('step')} "
            f"virtual_time={float(summary.get('virtual_time', 0.0)):.6f} "
            f"test_accuracy={_fmt(summary.get('test_accuracy'))} "
            f"test_macro_f1={_fmt(summary.get('test_macro_f1'))} "
            f"best_accuracy={_fmt(summary.get('best_accuracy'))} "
            f"mean_staleness={float(summary.get('mean_staleness', 0.0)):.6f} "
            f"seed={args.seed}"
        )


def _labels_from_tensor_dataset(dataset) -> np.ndarray:
    if not hasattr(dataset, "tensors") or len(dataset.tensors) < 2:
        raise RuntimeError("HHAR training dataset must expose label tensor as TensorDataset.tensors[1].")
    return np.asarray(dataset.tensors[1].detach().cpu().tolist(), dtype=np.int64)


def _make_runner(algorithm, clients, topology, mdfeel_topology, random_topology, graph, model_factory, test_loader, config):
    if algorithm == "clustered":
        if topology is None:
            raise ValueError("clustered algorithm requires topology.")
        return ClusteredTrainingRunner(clients, topology, model_factory, test_loader, config)
    if algorithm == "clustered_barrier":
        if topology is None:
            raise ValueError("clustered_barrier algorithm requires topology.")
        return ClusteredBarrierTrainingRunner(clients, topology, model_factory, test_loader, config)
    if algorithm == "clustered_no_mixing":
        if topology is None:
            raise ValueError("clustered_no_mixing algorithm requires topology.")
        return ClusteredTrainingRunner(clients, topology, model_factory, test_loader, config)
    if algorithm == "clustered_async":
        if topology is None:
            raise ValueError("clustered_async algorithm requires topology.")
        return ClusteredAsyncTrainingRunner(clients, topology, model_factory, test_loader, config)
    if algorithm == "clustered_asyncv1":
        if topology is None:
            raise ValueError("clustered_asyncv1 algorithm requires topology.")
        return ClusteredAsyncV1TrainingRunner(clients, topology, model_factory, test_loader, config)
    if algorithm == "clustered_async_clipprox":
        if topology is None:
            raise ValueError("clustered_async_clipprox algorithm requires topology.")
        return ClusteredAsyncClipProxTrainingRunner(clients, topology, model_factory, test_loader, config)
    if algorithm == "clustered_async_no_mixing":
        if topology is None:
            raise ValueError("clustered_async_no_mixing algorithm requires topology.")
        return ClusteredAsyncTrainingRunner(clients, topology, model_factory, test_loader, config)
    if algorithm == "random_clustered_async":
        if random_topology is None:
            raise ValueError("random_clustered_async algorithm requires random topology.")
        return ClusteredAsyncTrainingRunner(clients, random_topology, model_factory, test_loader, config)
    if algorithm == "mdfeel":
        if mdfeel_topology is None:
            raise ValueError("mdfeel algorithm requires adapted MD-FEEL topology.")
        return MDFeelTrainingRunner(clients, mdfeel_topology, model_factory, test_loader, config)
    if algorithm == "fedavg":
        return FedAvgRunner(clients, model_factory, test_loader, config)
    if algorithm == "dpsgd":
        if graph is None:
            raise ValueError("dpsgd algorithm requires initial feasible graph.")
        return DPSGDRunner(clients, graph, model_factory, test_loader, config)
    if algorithm == "adpsgd":
        if graph is None:
            raise ValueError("adpsgd algorithm requires initial feasible graph.")
        return ADPSGDRunner(clients, graph, model_factory, test_loader, config)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _partition_clients(args, labels, train_dataset=None):
    if args.dataset.lower() == "hhar" and args.hhar_client_partition == "group":
        if not hasattr(train_dataset, "hhar_group_ids"):
            raise RuntimeError("HHAR group client partition requires train_dataset.hhar_group_ids.")
        group_ids = np.asarray(train_dataset.hhar_group_ids.detach().cpu().tolist(), dtype=np.int64)
        grouped_clients = [
            np.where(group_ids == group_id)[0].astype(np.int64).tolist()
            for group_id in np.unique(group_ids)
        ]
        if not grouped_clients:
            raise RuntimeError("HHAR group client partition produced no clients.")
        return _merge_hhar_groups_to_clients(grouped_clients, args.num_clients, args.seed)
    if args.split == "iid":
        return partition_iid(labels, args.num_clients, args.samples_fraction, args.seed)
    return partition_dirichlet(labels, args.num_clients, args.dirichlet_alpha, args.samples_fraction, args.seed)


def _merge_hhar_groups_to_clients(grouped_clients: list[list[int]], num_clients: int, seed: int) -> list[list[int]]:
    if num_clients <= 0:
        raise ValueError("--num-clients must be positive.")
    if num_clients >= len(grouped_clients):
        return grouped_clients
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(grouped_clients)).tolist()
    bins: list[list[int]] = [[] for _ in range(num_clients)]
    bin_sizes = [0 for _ in range(num_clients)]
    for group_index in sorted(order, key=lambda idx: len(grouped_clients[idx]), reverse=True):
        target = min(range(num_clients), key=lambda idx: bin_sizes[idx])
        bins[target].extend(grouped_clients[group_index])
        bin_sizes[target] += len(grouped_clients[group_index])
    for indices in bins:
        rng.shuffle(indices)
    return bins


def _resolve_hhar_input_channels(args, train_dataset) -> int:
    if args.dataset.lower() != "hhar":
        return 3
    if args.hhar_input_channels > 0:
        return args.hhar_input_channels
    if hasattr(train_dataset, "tensors") and len(train_dataset.tensors) >= 1:
        features = train_dataset.tensors[0]
        if features.ndim == 3:
            return int(features.shape[1])
    return 3


def _build_topology(args, metadata, feasible_graph):
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
            feasible_graph=feasible_graph,
        )
    return build_deterministic_balanced_clusters(
        metadata,
        k=int(args.k),
        alpha_time=args.alpha_time,
        alpha_distribution=args.alpha_distribution,
        max_swaps=args.max_swaps,
        feasible_graph=feasible_graph,
    )


def _build_mdfeel_topology(args, metadata, feasible_graph):
    k_max = _resolve_k_max(args.k_max, len(metadata))
    if args.k.lower() == "auto":
        return select_best_mdfeel_topology(
            metadata,
            k_min=args.k_min,
            k_max=k_max,
            min_clients_per_cluster=args.min_clients_per_cluster,
            lambda_communication=args.lambda_communication,
            mixing_interval=args.mixing_interval,
            feasible_graph=feasible_graph,
            seed=args.seed,
        )
    return build_mdfeel_topology(
        metadata,
        k=int(args.k),
        feasible_graph=feasible_graph,
        seed=args.seed,
    )


def _resolve_fixed_or_auto_k(args, metadata, feasible_graph):
    if args.k.lower() == "auto":
        topology, _ = _build_topology(args, metadata, feasible_graph)
        return len(topology.clusters)
    return int(args.k)


def _build_random_topology(metadata, k: int, seed: int, feasible_graph, max_attempts: int):
    if max_attempts <= 0:
        raise ValueError("random_topology_attempts must be positive.")
    client_ids = [item.client_id for item in metadata]
    capacities = compute_capacities(len(metadata), k)
    clients_by_id = {client.client_id: client for client in metadata}
    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        shuffled = list(client_ids)
        rng.shuffle(shuffled)
        clusters = []
        offset = 0
        for capacity in capacities:
            clusters.append(sorted(shuffled[offset : offset + capacity]))
            offset += capacity
        leaders, leader_edges, leader_score, connected = select_connected_leaders(clusters, clients_by_id, feasible_graph)
        if not connected:
            continue
        topology = Topology(
            clusters=clusters,
            leaders=leaders,
            leader_edges=leader_edges,
            client_to_cluster={cid: cluster_idx for cluster_idx, cluster in enumerate(clusters) for cid in cluster},
        )
        metrics = {
            "K": k,
            "capacities": capacities,
            "leader_score": float(leader_score),
            "leader_edges": leader_edges,
            "connected": connected,
            "method": "random_balanced",
            "attempt": attempt,
            "attempts": max_attempts,
        }
        return topology, metrics
    raise RuntimeError("random topology failed to form connected leader overlay.")


def _maybe_subset_test(test_dataset, test_samples: int, seed: int):
    if test_samples <= 0 or test_samples >= len(test_dataset):
        return test_dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(np.arange(len(test_dataset)), size=test_samples, replace=False)).tolist()
    return Subset(test_dataset, indices)


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resolve_hhar_feature_mode(dataset: str, model: str, requested: str) -> str:
    if dataset.lower() != "hhar":
        return "stats"
    model_name = model.lower()
    value = requested.lower()
    if value == "auto":
        value = "raw" if model_name in {"hhar1dcnn", "hhar_1dcnn", "hhar_cnn1d", "cnn1d"} else "stats"
    if model_name in {"hhar1dcnn", "hhar_1dcnn", "hhar_cnn1d", "cnn1d"} and value != "raw":
        raise ValueError("--model hhar1dcnn requires --hhar-feature-mode raw.")
    if model_name in {"auto", "mlp", "flatten_mlp", "flattenmlp"} and value != "stats":
        raise ValueError("--model mlp/auto for HHAR requires --hhar-feature-mode stats.")
    return value


def _default_run_name(args, topology) -> str:
    k_part = f"_k{len(topology.clusters)}" if topology is not None else ""
    k_max_part = f"_kmax{args.k_max}" if str(args.k).lower() == "auto" else ""
    hhar_feature_part = f"_hhar{_resolve_hhar_feature_mode(args.dataset, args.model, args.hhar_feature_mode)}" if args.dataset.lower() == "hhar" else ""
    model_part = f"_model{args.model}{hhar_feature_part}"
    split_part = args.split
    if args.split == "dirichlet":
        split_part = f"{split_part}_a{args.dirichlet_alpha:g}"
    graph_part = f"{args.baseline_graph}_p{args.random_edge_prob:g}"
    algorithm_part = "_mdfeel" if args.algorithm == "mdfeel" else ""
    return (
        f"{args.dataset}_{split_part}_c{args.num_clients}{k_part}{k_max_part}{model_part}{algorithm_part}_"
        f"{graph_part}_mix{args.mixing_interval}_seed{args.seed}"
    )


def _algorithm_run_name(algorithm: str, args, active_mixing_interval: int) -> str:
    clipprox_part = ""
    if algorithm == "clustered_async_clipprox":
        clipprox_part = f"_mu{args.clipprox_mu:g}_clip{args.clipprox_clip_norm:g}"
    cluster_delta_clip_part = ""
    if algorithm == "clustered_asyncv1" and float(args.cluster_delta_clip_norm) > 0.0:
        cluster_delta_clip_part = f"_cdclip{args.cluster_delta_clip_norm:g}"
    hhar_feature_part = f"_hhar{_resolve_hhar_feature_mode(args.dataset, args.model, args.hhar_feature_mode)}" if args.dataset.lower() == "hhar" else ""
    return (
        f"{algorithm}_round{args.rounds}_event{args.events}_"
        f"mix{active_mixing_interval}_eval{args.eval_interval}_"
        f"le{args.local_epochs}_lr{args.lr:g}_bs{args.batch_size}_model{args.model}{hhar_feature_part}"
        f"{clipprox_part}{cluster_delta_clip_part}"
    )


def _resolve_k_max(k_max: str, num_clients: int) -> int | None:
    value = str(k_max).strip().lower()
    if value in {"none", "no", "null", "unbounded"}:
        return None
    if value == "sqrt":
        return max(1, int(math.floor(math.sqrt(num_clients))))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("--k-max must be an integer, `sqrt`, or `none`.") from exc
    if parsed <= 0:
        raise ValueError("--k-max must be positive when provided.")
    return parsed


def _prepare_run_dir(output_dir: Path, base_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / base_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _create_unique_run_dir(output_dir: Path, base_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(10000):
        suffix = "" if attempt == 0 else f"_rep{attempt:02d}"
        run_dir = output_dir / f"{base_name}{suffix}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique run directory for base name: {base_name}")


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.6f}"


def _is_algorithm_complete(algorithm_dir: Path) -> bool:
    if not (algorithm_dir / "metrics.csv").exists() or not (algorithm_dir / "final_summary.json").exists():
        return False
    if algorithm_dir.name.startswith("clustered") or algorithm_dir.name.startswith("mdfeel"):
        return (algorithm_dir / "cluster_counters.csv").exists()
    return True


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_shared_artifacts(
    shared_dir: Path,
    args,
    hhar_feature_mode: str,
    device: str,
    requested_run_name: str,
    run_dir: Path,
    metadata,
    topology,
    random_topology,
    cluster_metrics,
    mdfeel_topology,
    mdfeel_cluster_metrics,
    random_cluster_metrics,
    feasible_graph,
    run_algorithms: list[str],
) -> None:
    _save_json(
        shared_dir / "config.json",
        {
            **vars(args),
            "resolved_hhar_feature_mode": hhar_feature_mode,
            "device": device,
            "requested_run_name": requested_run_name,
            "resolved_run_name": run_dir.name,
            "layout": "shared_plus_algorithm_dirs",
        },
    )
    _save_json(shared_dir / "clients.json", metadata)
    if topology is not None:
        _save_json(shared_dir / "topology.json", topology)
    if mdfeel_topology is not None:
        _save_json(shared_dir / "mdfeel_topology.json", mdfeel_topology)
    if random_topology is not None:
        _save_json(shared_dir / "random_topology.json", random_topology)
    if cluster_metrics is not None:
        _save_json(shared_dir / "cluster_metrics.json", cluster_metrics)
        if "k_search" in cluster_metrics:
            _save_json(shared_dir / "k_search.json", cluster_metrics["k_search"])
    if mdfeel_cluster_metrics is not None:
        _save_json(shared_dir / "mdfeel_cluster_metrics.json", mdfeel_cluster_metrics)
        if "k_search" in mdfeel_cluster_metrics:
            _save_json(shared_dir / "mdfeel_k_search.json", mdfeel_cluster_metrics["k_search"])
    if random_cluster_metrics is not None:
        _save_json(shared_dir / "random_cluster_metrics.json", random_cluster_metrics)
    if feasible_graph is not None:
        _save_json(
            shared_dir / "initial_feasible_graph.json",
            {
                "edges": graph_edges(feasible_graph),
                "graph_type": args.baseline_graph,
                "random_edge_prob": args.random_edge_prob,
                "seed": args.seed,
                "shared_by": [
                    algorithm for algorithm in run_algorithms if algorithm in GRAPHED_ALGORITHMS
                ],
            },
        )


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    main()
