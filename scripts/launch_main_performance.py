from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_SCRIPT = REPO_ROOT / "scripts" / "run_training_simulation.py"


@dataclass
class Job:
    dataset: str
    alpha: float
    seed: int
    num_clients: int
    algorithm: str
    mixing_interval: int
    command: list[str]
    log_path: Path


@dataclass
class RunningJob:
    job: Job
    gpu: str
    process: subprocess.Popen
    log_file: object
    log_offset: int = 0
    last_progress: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch main-performance training runs across multiple GPUs."
    )
    # Resource scheduling:
    #   --gpus: comma-separated physical GPU ids, e.g. "0", "0,1,2", "0,1,2,3".
    #   --jobs-per-gpu: number of subprocess jobs launched concurrently per GPU.
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids, e.g. 0,1,2,3.")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)

    # Experiment grid:
    #   --datasets: comma-separated; run_training_simulation.py supports
    #       synthetic,mnist,fashionmnist,cifar10,cifar100,svhn,hhar.
    #   --alphas: comma-separated Dirichlet alpha values; smaller means more non-IID.
    #   --seeds: comma-separated random seeds.
    parser.add_argument("--datasets", default="cifar10")
    parser.add_argument("--alphas", default="0.1,0.3,0.5")
    parser.add_argument("--seeds", default="0,1,2")

    # Algorithm:
    #   choices supported downstream:
    #       clustered, clustered_barrier, clustered_no_mixing, clustered_async,
    #       clustered_asyncv1,
    #       clustered_async_no_mixing, random_clustered_async,
    #       fedavg, dpsgd, adpsgd, mdfeel, core3, all.
    #   Use "all" for fair same-run comparison with a shared initial feasible graph.
    parser.add_argument("--algorithm", choices=["clustered", "clustered_barrier", "clustered_no_mixing", "clustered_async", "clustered_asyncv1", "clustered_async_clipprox", "clustered_async_no_mixing", "random_clustered_async", "fedavg", "dpsgd", "adpsgd", "mdfeel", "core3", "all"], default="all")
    parser.add_argument(
        "--algorithm-plan",
        default="",
        help=(
            "Comma-separated algorithm:mixing_interval entries, e.g. "
            "clustered_async:1,clustered_async_no_mixing:1,clustered:1. "
            "When provided, all entries share one GPU scheduling queue."
        ),
    )

    # Data partitioning:
    #   --split choices: iid, dirichlet.
    parser.add_argument(
        "--num-clients",
        default="50",
        help="Number of clients, or comma-separated client counts such as 50,100,200.",
    )
    parser.add_argument("--split", choices=["iid", "dirichlet"], default="dirichlet")

    # Clustering:
    #   --k: fixed integer cluster count or "auto".
    #   --k-max: integer upper bound, "sqrt", or "none".
    parser.add_argument("--k", default="auto")
    parser.add_argument("--k-max", default="sqrt")

    # Training length:
    #   --rounds controls synchronous methods: fedavg, dpsgd, clustered.
    #   --events controls asynchronous methods: clustered_async, adpsgd.
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--events", type=int, default=500)

    # Communication/evaluation:
    #   --mixing-interval: mix every N local updates; use 0 to disable where supported.
    #   --eval-interval: evaluate every N rounds/events.
    parser.add_argument("--mixing-interval", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--sync-comm-bandwidth-proxy", type=float, default=1_000_000.0)

    # Shared baseline graph:
    #   --baseline-graph choices: full, ring, random.
    #   --random-edge-prob is only used when --baseline-graph=random.
    parser.add_argument("--baseline-graph", choices=["full", "ring", "random"], default="random")
    parser.add_argument("--random-edge-prob", type=float, default=0.05)

    # Paths and execution mode.
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/training")
    parser.add_argument("--log-dir", default="results/logs/main_performance")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--cluster-delta-clip-norm", type=float, default=0.0)
    parser.add_argument("--model", default="auto", choices=["auto", "mlp", "cnnbn", "resnet20", "hhar1dcnn"])
    parser.add_argument("--hhar-feature-mode", default="auto", choices=["auto", "stats", "raw"])
    parser.add_argument("--hhar-window-size", type=int, default=128)
    parser.add_argument("--hhar-step-size", type=int, default=64)
    parser.add_argument("--hhar-sensor-mode", choices=["auto", "acc", "acc_gyro"], default="auto")
    parser.add_argument("--hhar-test-split", choices=["window", "group"], default="window")
    parser.add_argument("--hhar-input-channels", type=int, default=0)
    parser.add_argument("--hhar-client-partition", choices=["dirichlet", "group"], default="dirichlet")
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    gpus = _resolve_gpus(args.gpus)
    slots = deque(gpu for _ in range(args.jobs_per_gpu) for gpu in gpus)
    jobs = deque(_build_jobs(args, extra_args))
    if not jobs:
        raise SystemExit("No jobs to run.")

    print(
        f"[launcher] jobs={len(jobs)} gpus={','.join(gpus)} "
        f"jobs_per_gpu={args.jobs_per_gpu}",
        flush=True,
    )
    if args.dry_run:
        for job in jobs:
            print(f"[dry-run] {' '.join(job.command)}")
        return

    running: list[RunningJob] = []
    failures: list[tuple[Job, int]] = []
    while jobs or running:
        while jobs and slots:
            gpu = slots.popleft()
            job = jobs.popleft()
            running.append(_start_job(job, gpu))

        time.sleep(5)
        still_running = []
        for item in running:
            returncode = item.process.poll()
            if returncode is None:
                _print_new_progress(item)
                still_running.append(item)
                continue
            item.log_file.close()
            slots.append(item.gpu)
            status = "done" if returncode == 0 else f"failed rc={returncode}"
            print(
                f"[launcher] {status} gpu={item.gpu} "
                f"algorithm={item.job.algorithm} mix={item.job.mixing_interval} "
                f"dataset={item.job.dataset} alpha={item.job.alpha:g} "
                f"clients={item.job.num_clients} seed={item.job.seed} "
                f"log={item.job.log_path}",
                flush=True,
            )
            if returncode != 0:
                failures.append((item.job, returncode))
        running = still_running

    if failures:
        print(f"[launcher] failures={len(failures)}", flush=True)
        for job, returncode in failures:
            print(
                f"[launcher] failed rc={returncode} algorithm={job.algorithm} "
                f"mix={job.mixing_interval} dataset={job.dataset} "
                f"alpha={job.alpha:g} clients={job.num_clients} "
                f"seed={job.seed} log={job.log_path}",
                flush=True,
            )
        raise SystemExit(1)
    print("[launcher] all jobs completed.", flush=True)


def _build_jobs(args, extra_args: list[str]) -> list[Job]:
    datasets = _split_csv(args.datasets, str)
    alphas = _split_csv(args.alphas, float)
    seeds = _split_csv(args.seeds, int)
    num_clients_values = _split_csv(str(args.num_clients), int)
    algorithm_plan = _parse_algorithm_plan(args)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    used_log_paths: set[Path] = set()
    for algorithm, mixing_interval in algorithm_plan:
        for dataset in datasets:
            for alpha in alphas:
                for num_clients in num_clients_values:
                    for seed in seeds:
                        command = [
                            sys.executable,
                            str(TRAINING_SCRIPT),
                            "--algorithm",
                            algorithm,
                            "--dataset",
                            dataset,
                            "--data-dir",
                            args.data_dir,
                            "--num-clients",
                            str(num_clients),
                            "--split",
                            args.split,
                            "--k",
                            args.k,
                            "--k-max",
                            args.k_max,
                            "--rounds",
                            str(args.rounds),
                            "--events",
                            str(args.events),
                            "--mixing-interval",
                            str(mixing_interval),
                            "--sync-comm-bandwidth-proxy",
                            str(args.sync_comm_bandwidth_proxy),
                            "--eval-interval",
                            str(args.eval_interval),
                            "--baseline-graph",
                            args.baseline_graph,
                            "--random-edge-prob",
                            str(args.random_edge_prob),
                            "--seed",
                            str(seed),
                            "--output-dir",
                            args.output_dir,
                            "--device",
                            args.device,
                            "--batch-size",
                            str(args.batch_size),
                            "--test-samples",
                            str(args.test_samples),
                            "--local-epochs",
                            str(args.local_epochs),
                            "--lr",
                            str(args.lr),
                            "--momentum",
                            str(args.momentum),
                            "--weight-decay",
                            str(args.weight_decay),
                            "--lr-scheduler",
                            args.lr_scheduler,
                            "--cluster-delta-clip-norm",
                            str(args.cluster_delta_clip_norm),
                            "--model",
                            args.model,
                            "--hhar-feature-mode",
                            args.hhar_feature_mode,
                            "--hhar-window-size",
                            str(args.hhar_window_size),
                            "--hhar-step-size",
                            str(args.hhar_step_size),
                            "--hhar-sensor-mode",
                            args.hhar_sensor_mode,
                            "--hhar-test-split",
                            args.hhar_test_split,
                            "--hhar-input-channels",
                            str(args.hhar_input_channels),
                            "--hhar-client-partition",
                            args.hhar_client_partition,
                        ]
                        if args.split == "dirichlet":
                            command.extend(["--dirichlet-alpha", str(alpha)])
                        command.extend(extra_args)
                        log_stem = f"{dataset}_a{alpha:g}_c{num_clients}_seed{seed}_{algorithm}_mix{mixing_interval}"
                        jobs.append(
                            Job(
                                dataset=dataset,
                                alpha=alpha,
                                seed=seed,
                                num_clients=num_clients,
                                algorithm=algorithm,
                                mixing_interval=mixing_interval,
                                command=command,
                                log_path=_unique_log_path(log_dir, log_stem, used_log_paths),
                            )
                        )
    return jobs


def _start_job(job: Job, gpu: str) -> RunningJob:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = job.log_path.open("w", encoding="utf-8")
    print(
        f"[launcher] start gpu={gpu} algorithm={job.algorithm} mix={job.mixing_interval} "
        f"dataset={job.dataset} alpha={job.alpha:g} clients={job.num_clients} "
        f"seed={job.seed} log={job.log_path}",
        flush=True,
    )
    process = subprocess.Popen(
        job.command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningJob(job=job, gpu=gpu, process=process, log_file=log_file)


def _print_new_progress(item: RunningJob) -> None:
    try:
        with item.job.log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(item.log_offset)
            chunk = handle.read()
            item.log_offset = handle.tell()
    except OSError:
        return

    progress_lines = [
        line.strip()
        for line in chunk.splitlines()
        if "[progress]" in line
        or "[training]" in line
        or "[hhar]" in line
    ]
    if not progress_lines:
        return
    latest = progress_lines[-1]
    if latest == item.last_progress:
        return
    item.last_progress = latest
    print(
        f"[launcher][progress] gpu={item.gpu} "
        f"algorithm={item.job.algorithm} mix={item.job.mixing_interval} "
        f"dataset={item.job.dataset} alpha={item.job.alpha:g} "
        f"clients={item.job.num_clients} seed={item.job.seed} "
        f"{latest}",
        flush=True,
    )


def _unique_log_path(log_dir: Path, stem: str, reserved: set[Path]) -> Path:
    for attempt in range(10000):
        suffix = "" if attempt == 0 else f"_rep{attempt:02d}"
        path = log_dir / f"{stem}{suffix}.log"
        if path not in reserved and not path.exists():
            reserved.add(path)
            return path
    raise RuntimeError(f"Could not create a unique log path for base name: {stem}")


def _resolve_gpus(gpus_arg: str | None) -> list[str]:
    if gpus_arg:
        gpus = [item.strip() for item in gpus_arg.split(",") if item.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpus = [item.strip() for item in visible.split(",") if item.strip()] if visible else ["0"]
    if not gpus:
        raise SystemExit("No GPUs selected. Pass --gpus 0,1 or set CUDA_VISIBLE_DEVICES.")
    return gpus


def _parse_algorithm_plan(args) -> list[tuple[str, int]]:
    valid_algorithms = {
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
    }
    if not args.algorithm_plan:
        return [(args.algorithm, args.mixing_interval)]
    plan = []
    for item in _split_csv(args.algorithm_plan, str):
        if ":" not in item:
            raise SystemExit(f"Invalid --algorithm-plan entry {item!r}; expected algorithm:mixing_interval.")
        algorithm, interval_text = item.split(":", 1)
        algorithm = algorithm.strip()
        if algorithm not in valid_algorithms:
            raise SystemExit(f"Invalid algorithm in --algorithm-plan: {algorithm!r}.")
        try:
            mixing_interval = int(interval_text)
        except ValueError as exc:
            raise SystemExit(f"Invalid mixing interval in --algorithm-plan entry {item!r}.") from exc
        plan.append((algorithm, mixing_interval))
    if not plan:
        raise SystemExit("--algorithm-plan did not contain any entries.")
    return plan


def _split_csv(value: str, caster):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise SystemExit(f"Empty comma-separated value: {value!r}")
    return [caster(item) for item in items]


if __name__ == "__main__":
    main()
