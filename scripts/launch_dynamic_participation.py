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
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_dynamic_participation.py"


@dataclass
class Job:
    policy: str
    departure_ratio: float
    seed: int
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
    parser = argparse.ArgumentParser(description="Launch Figure 5 dynamic-participation experiments.")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--seeds", default="0,3,7")
    parser.add_argument("--departure-ratios", default="0.2,0.4,0.6")
    parser.add_argument("--refresh-policies", default="blade_refresh,stale_topology,full_reconstruction")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--alphas", default="0.3")
    parser.add_argument("--num-clients", type=int, default=50)
    parser.add_argument("--pool-clients", type=int, default=80)
    parser.add_argument("--events", type=int, default=2100)
    parser.add_argument("--leave-event", type=int, default=700)
    parser.add_argument("--join-event", type=int, default=1400)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--split", choices=["iid", "dirichlet"], default="dirichlet")
    parser.add_argument("--k", default="auto")
    parser.add_argument("--k-max", default="sqrt")
    parser.add_argument("--mixing-interval", type=int, default=1)
    parser.add_argument("--baseline-graph", choices=["full", "ring", "random"], default="random")
    parser.add_argument("--random-edge-prob", type=float, default=0.45)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/training_figure5_dynamic_cifar10_a03")
    parser.add_argument("--log-dir", default="results/logs_figure5_dynamic_cifar10_a03")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="resnet20")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--bytes-per-time-unit", type=float, default=1_000_000.0)
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    gpus = _resolve_gpus(args.gpus)
    slots = deque(gpu for _ in range(args.jobs_per_gpu) for gpu in gpus)
    jobs = deque(_build_jobs(args, extra_args))
    print(
        f"[dynamic-launcher] jobs={len(jobs)} gpus={','.join(gpus)} "
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
                f"[dynamic-launcher] {status} gpu={item.gpu} "
                f"policy={item.job.policy} ratio={item.job.departure_ratio:g} "
                f"seed={item.job.seed} log={item.job.log_path}",
                flush=True,
            )
            if returncode != 0:
                failures.append((item.job, returncode))
        running = still_running

    if failures:
        print(f"[dynamic-launcher] failures={len(failures)}", flush=True)
        for job, returncode in failures:
            print(
                f"[dynamic-launcher] failed rc={returncode} policy={job.policy} "
                f"ratio={job.departure_ratio:g} seed={job.seed} log={job.log_path}",
                flush=True,
            )
        raise SystemExit(1)
    print("[dynamic-launcher] all jobs completed.", flush=True)


def _build_jobs(args, extra_args: list[str]) -> list[Job]:
    seeds = _split_csv(args.seeds, int)
    ratios = _split_csv(args.departure_ratios, float)
    policies = _split_csv(args.refresh_policies, str)
    alphas = _split_csv(args.alphas, float)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    used_logs: set[Path] = set()
    for policy in policies:
        for alpha in alphas:
            for ratio in ratios:
                for seed in seeds:
                    command = [
                        sys.executable,
                        str(RUN_SCRIPT),
                        "--dataset",
                        args.dataset,
                        "--data-dir",
                        args.data_dir,
                        "--num-clients",
                        str(args.num_clients),
                        "--pool-clients",
                        str(args.pool_clients),
                        "--split",
                        args.split,
                        "--dirichlet-alpha",
                        str(alpha),
                        "--departure-ratio",
                        str(ratio),
                        "--refresh-policy",
                        policy,
                        "--leave-event",
                        str(args.leave_event),
                        "--join-event",
                        str(args.join_event),
                        "--events",
                        str(args.events),
                        "--eval-interval",
                        str(args.eval_interval),
                        "--seed",
                        str(seed),
                        "--k",
                        args.k,
                        "--k-max",
                        args.k_max,
                        "--mixing-interval",
                        str(args.mixing_interval),
                        "--baseline-graph",
                        args.baseline_graph,
                        "--random-edge-prob",
                        str(args.random_edge_prob),
                        "--output-dir",
                        args.output_dir,
                        "--device",
                        args.device,
                        "--model",
                        args.model,
                        "--batch-size",
                        str(args.batch_size),
                        "--test-samples",
                        str(args.test_samples),
                        "--local-epochs",
                        str(args.local_epochs),
                        "--lr",
                        str(args.lr),
                        "--lr-scheduler",
                        args.lr_scheduler,
                        "--momentum",
                        str(args.momentum),
                        "--weight-decay",
                        str(args.weight_decay),
                        "--bytes-per-time-unit",
                        str(args.bytes_per_time_unit),
                        *extra_args,
                    ]
                    log_name = f"{args.dataset}_a{alpha:g}_{policy}_r{ratio:g}_seed{seed}.log"
                    log_path = _unique_log_path(log_dir / log_name, used_logs)
                    used_logs.add(log_path)
                    jobs.append(Job(policy=policy, departure_ratio=ratio, seed=seed, command=command, log_path=log_path))
    return jobs


def _start_job(job: Job, gpu: str) -> RunningJob:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = job.log_path.open("w", encoding="utf-8")
    print(
        f"[dynamic-launcher] start gpu={gpu} policy={job.policy} "
        f"ratio={job.departure_ratio:g} seed={job.seed} log={job.log_path}",
        flush=True,
    )
    process = subprocess.Popen(job.command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    return RunningJob(job=job, gpu=gpu, process=process, log_file=log_file)


def _print_new_progress(item: RunningJob) -> None:
    try:
        with item.job.log_path.open("r", encoding="utf-8", errors="ignore") as f:
            f.seek(item.log_offset)
            data = f.read()
            item.log_offset = f.tell()
    except OSError:
        return
    for line in data.splitlines():
        if line.startswith("[progress]") or line.startswith("[dynamic]"):
            item.last_progress = line
    if item.last_progress:
        print(
            f"[dynamic-launcher] gpu={item.gpu} policy={item.job.policy} "
            f"ratio={item.job.departure_ratio:g} seed={item.job.seed} {item.last_progress}",
            flush=True,
        )


def _split_csv(value: str, caster):
    return [caster(item.strip()) for item in value.split(",") if item.strip()]


def _resolve_gpus(value: str | None) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    return ["0"]


def _unique_log_path(base: Path, used: set[Path]) -> Path:
    if base not in used and not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    for idx in range(1, 10000):
        candidate = base.with_name(f"{stem}_rep{idx:02d}{suffix}")
        if candidate not in used and not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique log path for {base}")


if __name__ == "__main__":
    main()
