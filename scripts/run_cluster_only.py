from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Allow running via `python scripts/run_cluster_only.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clustered_dfl.clustering import build_deterministic_balanced_clusters, select_best_k_topology
from clustered_dfl.data import build_client_metadata, load_labels, partition_dirichlet, partition_iid


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic balanced clustering only.")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "mnist", "fashionmnist", "cifar10", "cifar100", "svhn"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--split", choices=["iid", "dirichlet"], default="dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--samples-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", required=True, help="Fixed K integer or `auto`.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--min-clients-per-cluster", type=int, default=2)
    parser.add_argument("--alpha-time", type=float, default=0.5)
    parser.add_argument("--alpha-distribution", type=float, default=0.5)
    parser.add_argument("--lambda-communication", type=float, default=0.2)
    parser.add_argument("--model-size", type=float, default=1.0)
    parser.add_argument("--mixing-interval", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    parser.add_argument("--epsilon-swap", type=float, default=1.0e-12)
    parser.add_argument("--max-swaps", type=int, default=10)
    parser.add_argument("--time-mode", choices=["num_samples", "local_epochs_num_samples"], default="num_samples")
    parser.add_argument("--local-epochs", type=float, default=1.0)
    parser.add_argument("--output-dir", default="results/cluster_only")
    args = parser.parse_args()

    _set_seed(args.seed)

    labels, num_classes = load_labels(args.dataset, args.data_dir)
    if args.split == "iid":
        client_indices = partition_iid(
            labels,
            num_clients=args.num_clients,
            samples_fraction=args.samples_fraction,
            seed=args.seed,
        )
    else:
        client_indices = partition_dirichlet(
            labels,
            num_clients=args.num_clients,
            alpha=args.dirichlet_alpha,
            samples_fraction=args.samples_fraction,
            seed=args.seed,
        )

    clients = build_client_metadata(
        client_indices,
        labels,
        num_classes,
        time_mode=args.time_mode,
        local_epochs=args.local_epochs,
    )
    if args.k.lower() == "auto":
        topology, metrics = select_best_k_topology(
            clients,
            k_min=args.k_min,
            min_clients_per_cluster=args.min_clients_per_cluster,
            alpha_time=args.alpha_time,
            alpha_distribution=args.alpha_distribution,
            lambda_communication=args.lambda_communication,
            model_size=args.model_size,
            mixing_interval=args.mixing_interval,
            epsilon=args.epsilon,
            epsilon_swap=args.epsilon_swap,
            max_swaps=args.max_swaps,
        )
        active_k = metrics["K"]
    else:
        active_k = int(args.k)
        topology, metrics = build_deterministic_balanced_clusters(
            clients,
            k=active_k,
            alpha_time=args.alpha_time,
            alpha_distribution=args.alpha_distribution,
            epsilon=args.epsilon,
            epsilon_swap=args.epsilon_swap,
            max_swaps=args.max_swaps,
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}_{args.split}_c{args.num_clients}_k{args.k}_selected{active_k}_{stamp}"
    run_dir = Path(args.output_dir) / run_name
    _save_json(run_dir / "clients.json", clients)
    _save_json(run_dir / "topology.json", topology)
    _save_json(run_dir / "cluster_metrics.json", metrics)
    if "k_search" in metrics:
        _save_json(run_dir / "k_search.json", metrics["k_search"])
    _save_json(run_dir / "config.json", vars(args))

    print(f"[cluster] output_dir={run_dir}")
    print(
        "[cluster] "
        f"K={metrics['K']} "
        f"J_T={metrics['waiting_cost']:.6f} "
        f"Jbar_T={metrics['normalized_waiting_cost']:.6f} "
        f"J_D={metrics['distribution_cost']:.6f} "
        f"Jbar_D={metrics['normalized_distribution_cost']:.6f} "
        f"J_bal={metrics['balanced_score']:.6f} "
        f"leader_edges={len(topology.leader_edges)} "
        f"swaps={metrics['accepted_swaps']} "
        f"seed={args.seed}"
    )


if __name__ == "__main__":
    main()
