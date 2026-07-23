from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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
from clustered_dfl.topology import build_complete_graph, select_connected_leaders
from clustered_dfl.training import (
    ADPSGDRunner,
    ClusteredAsyncTrainingRunner,
    ClusteredTrainingRunner,
    DPSGDRunner,
    FedAvgRunner,
    TrainingConfig,
)
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
    "best_accuracy",
    "mean_staleness",
    "max_staleness",
    "leader_edges",
    "transmitted_bytes_proxy",
    "model_divergence",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run client training simulation for clustered, D-PSGD, and AD-PSGD.")
    parser.add_argument(
        "--algorithm",
        choices=[
            "clustered",
            "clustered_no_mixing",
            "clustered_async",
            "clustered_async_no_mixing",
            "random_clustered_async",
            "fedavg",
            "dpsgd",
            "adpsgd",
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=1000, help="Use 0 for the full test set.")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--k", default="auto", help="Clustered algorithms only: fixed integer K or `auto`.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--min-clients-per-cluster", type=int, default=2)
    parser.add_argument("--alpha-time", type=float, default=0.5)
    parser.add_argument("--alpha-distribution", type=float, default=0.5)
    parser.add_argument("--lambda-communication", type=float, default=0.2)
    parser.add_argument("--mixing-interval", type=int, default=5)
    parser.add_argument(
        "--max-async-update-skew",
        type=int,
        default=1,
        help="AD-PSGD fairness bound: a client can be at most this many updates ahead of the slowest client; use -1 to disable.",
    )
    parser.add_argument("--max-swaps", type=int, default=10)
    parser.add_argument("--baseline-graph", choices=["full", "ring", "random"], default="random")
    parser.add_argument("--random-edge-prob", type=float, default=0.3)
    parser.add_argument("--output-dir", default="results/training")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    print(
        f"[training] start dataset={args.dataset} algorithm={args.algorithm} "
        f"num_clients={args.num_clients} seed={args.seed}",
        flush=True,
    )
    if args.dataset.lower() == "hhar":
        print("[training] loading HHAR dataset once for labels and tensors...", flush=True)
        train_dataset, test_dataset, loaded_num_classes = load_torch_datasets(args.dataset, args.data_dir, seed=args.seed)
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
    test_loader = make_test_loader(test_dataset, args.batch_size, args.num_workers)

    print(f"[training] partitioning {len(labels)} samples with split={args.split}...", flush=True)
    client_indices = _partition_clients(args, labels)
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
    random_topology = None
    random_cluster_metrics = None
    needs_balanced_topology = args.algorithm in {"clustered", "clustered_no_mixing", "clustered_async", "clustered_async_no_mixing", "all"}
    needs_random_topology = args.algorithm in {"random_clustered_async", "all"}
    if needs_balanced_topology:
        print(f"[training] building balanced topology k={args.k}...", flush=True)
        topology, cluster_metrics = _build_topology(args, metadata)
        print(f"[training] balanced topology ready K={len(topology.clusters)}.", flush=True)
    if needs_random_topology:
        active_k = len(topology.clusters) if topology is not None else _resolve_fixed_or_auto_k(args, metadata)
        print(f"[training] building random topology K={active_k}...", flush=True)
        random_topology, random_cluster_metrics = _build_random_topology(metadata, active_k, args.seed)

    graph = None
    if args.algorithm in {"dpsgd", "adpsgd", "all"}:
        print(
            f"[training] building shared D-PSGD/AD-PSGD baseline graph "
            f"mode={args.baseline_graph}...",
            flush=True,
        )
        graph = build_client_graph(client_ids, args.baseline_graph, args.random_edge_prob, args.seed)

    model_factory = lambda: create_model(args.dataset, hidden_dim=args.hidden_dim)
    run_algorithms = (
        ["clustered_async", "fedavg", "dpsgd", "adpsgd", "clustered_async_no_mixing", "random_clustered_async"]
        if args.algorithm == "all"
        else [args.algorithm]
    )
    all_metrics: list[dict] = []
    summaries: dict[str, dict] = {}
    for algorithm in run_algorithms:
        print(f"[training] running {algorithm}...", flush=True)
        _set_seed(args.seed)
        clients = build_client_states(
            train_dataset,
            client_indices,
            metadata,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        active_mixing_interval = 0 if algorithm in {"clustered_no_mixing", "clustered_async_no_mixing"} else args.mixing_interval
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
        )
        runner = _make_runner(algorithm, clients, topology, random_topology, graph, model_factory, test_loader, config)
        artifacts = runner.run()
        all_metrics.extend(artifacts.metrics)
        summaries[algorithm] = artifacts.summary
        print(f"[training] finished {algorithm}.", flush=True)

    run_name = args.run_name or _default_run_name(args, topology)
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_json(run_dir / "config.json", {**vars(args), "device": device})
    _save_json(run_dir / "clients.json", metadata)
    if topology is not None:
        _save_json(run_dir / "topology.json", topology)
    if random_topology is not None:
        _save_json(run_dir / "random_topology.json", random_topology)
    if cluster_metrics is not None:
        _save_json(run_dir / "cluster_metrics.json", cluster_metrics)
        if "k_search" in cluster_metrics:
            _save_json(run_dir / "k_search.json", cluster_metrics["k_search"])
    if random_cluster_metrics is not None:
        _save_json(run_dir / "random_cluster_metrics.json", random_cluster_metrics)
    if graph is not None:
        _save_json(
            run_dir / "baseline_graph.json",
            {
                "edges": graph_edges(graph),
                "graph_type": args.baseline_graph,
                "random_edge_prob": args.random_edge_prob,
                "seed": args.seed,
                "shared_by": [
                    algorithm for algorithm in run_algorithms if algorithm in {"dpsgd", "adpsgd"}
                ],
            },
        )
    _write_metrics_csv(run_dir / "metrics.csv", all_metrics)
    _save_json(run_dir / "final_summary.json", summaries if args.algorithm == "all" else summaries[args.algorithm])

    print(f"[training] output_dir={run_dir}")
    for algorithm, summary in summaries.items():
        print(
            "[training] "
            f"algorithm={algorithm} "
            f"step={summary.get('step')} "
            f"virtual_time={float(summary.get('virtual_time', 0.0)):.6f} "
            f"test_accuracy={_fmt(summary.get('test_accuracy'))} "
            f"best_accuracy={_fmt(summary.get('best_accuracy'))} "
            f"mean_staleness={float(summary.get('mean_staleness', 0.0)):.6f} "
            f"seed={args.seed}"
        )


def _labels_from_tensor_dataset(dataset) -> np.ndarray:
    if not hasattr(dataset, "tensors") or len(dataset.tensors) < 2:
        raise RuntimeError("HHAR training dataset must expose label tensor as TensorDataset.tensors[1].")
    return dataset.tensors[1].detach().cpu().numpy().astype(np.int64)


def _make_runner(algorithm, clients, topology, random_topology, graph, model_factory, test_loader, config):
    if algorithm == "clustered":
        if topology is None:
            raise ValueError("clustered algorithm requires topology.")
        return ClusteredTrainingRunner(clients, topology, model_factory, test_loader, config)
    if algorithm == "clustered_no_mixing":
        if topology is None:
            raise ValueError("clustered_no_mixing algorithm requires topology.")
        return ClusteredTrainingRunner(clients, topology, model_factory, test_loader, config)
    if algorithm == "clustered_async":
        if topology is None:
            raise ValueError("clustered_async algorithm requires topology.")
        return ClusteredAsyncTrainingRunner(clients, topology, model_factory, test_loader, config)
    if algorithm == "clustered_async_no_mixing":
        if topology is None:
            raise ValueError("clustered_async_no_mixing algorithm requires topology.")
        return ClusteredAsyncTrainingRunner(clients, topology, model_factory, test_loader, config)
    if algorithm == "random_clustered_async":
        if random_topology is None:
            raise ValueError("random_clustered_async algorithm requires random topology.")
        return ClusteredAsyncTrainingRunner(clients, random_topology, model_factory, test_loader, config)
    if algorithm == "fedavg":
        return FedAvgRunner(clients, model_factory, test_loader, config)
    if algorithm == "dpsgd":
        if graph is None:
            raise ValueError("dpsgd algorithm requires baseline graph.")
        return DPSGDRunner(clients, graph, model_factory, test_loader, config)
    if algorithm == "adpsgd":
        if graph is None:
            raise ValueError("adpsgd algorithm requires baseline graph.")
        return ADPSGDRunner(clients, graph, model_factory, test_loader, config)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _partition_clients(args, labels):
    if args.split == "iid":
        return partition_iid(labels, args.num_clients, args.samples_fraction, args.seed)
    return partition_dirichlet(labels, args.num_clients, args.dirichlet_alpha, args.samples_fraction, args.seed)


def _build_topology(args, metadata):
    if args.k.lower() == "auto":
        return select_best_k_topology(
            metadata,
            k_min=args.k_min,
            min_clients_per_cluster=args.min_clients_per_cluster,
            alpha_time=args.alpha_time,
            alpha_distribution=args.alpha_distribution,
            lambda_communication=args.lambda_communication,
            mixing_interval=args.mixing_interval,
            max_swaps=args.max_swaps,
        )
    return build_deterministic_balanced_clusters(
        metadata,
        k=int(args.k),
        alpha_time=args.alpha_time,
        alpha_distribution=args.alpha_distribution,
        max_swaps=args.max_swaps,
    )


def _resolve_fixed_or_auto_k(args, metadata):
    if args.k.lower() == "auto":
        topology, _ = _build_topology(args, metadata)
        return len(topology.clusters)
    return int(args.k)


def _build_random_topology(metadata, k: int, seed: int):
    rng = random.Random(seed)
    client_ids = [item.client_id for item in metadata]
    shuffled = list(client_ids)
    rng.shuffle(shuffled)
    capacities = compute_capacities(len(metadata), k)
    clusters = []
    offset = 0
    for capacity in capacities:
        clusters.append(sorted(shuffled[offset : offset + capacity]))
        offset += capacity
    clients_by_id = {client.client_id: client for client in metadata}
    graph = build_complete_graph(sorted(client_ids))
    leaders, leader_edges, leader_score, connected = select_connected_leaders(clusters, clients_by_id, graph)
    if not connected:
        raise RuntimeError("random topology failed to form connected leader overlay.")
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
    }
    return topology, metrics


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


def _default_run_name(args, topology) -> str:
    k_part = f"_k{len(topology.clusters)}" if topology is not None else ""
    return f"{args.dataset}_{args.split}_c{args.num_clients}{k_part}_{args.algorithm}_round{args.rounds}"


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.6f}"


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


if __name__ == "__main__":
    main()
