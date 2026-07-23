from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize main-performance runs for Fig.1/Table1.")
    parser.add_argument("--results-dir", default="results/training")
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--output-dir", default="results/summary_main_performance")
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(Path(args.results_dir).glob("*")):
        metrics_path = run_dir / "metrics.csv"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        metrics = _read_metrics(metrics_path)
        for algorithm, algo_rows in _group_by_algorithm(metrics).items():
            final = algo_rows[-1]
            target = _first_target_row(algo_rows, args.target_accuracy)
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "dataset": config.get("dataset", ""),
                    "alpha": config.get("dirichlet_alpha", ""),
                    "split": config.get("split", ""),
                    "num_clients": config.get("num_clients", ""),
                    "seed": config.get("seed", ""),
                    "algorithm": algorithm,
                    "final_accuracy": final.get("test_accuracy", ""),
                    "final_loss": final.get("test_loss", ""),
                    "final_virtual_time": final.get("virtual_time", ""),
                    "final_transmitted_bytes": final.get("transmitted_bytes_proxy", ""),
                    "target_accuracy": args.target_accuracy if args.target_accuracy is not None else "",
                    "time_to_target": target.get("virtual_time", "") if target else "",
                    "bytes_to_target": target.get("transmitted_bytes_proxy", "") if target else "",
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_run_summary.csv", rows)
    _write_csv(output_dir / "aggregate_summary.csv", _aggregate(rows))
    print(f"[summary] output_dir={output_dir}")
    print(f"[summary] runs={len(rows)} target_accuracy={args.target_accuracy}")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _group_by_algorithm(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("algorithm", "")].append(row)
    for algo_rows in grouped.values():
        algo_rows.sort(key=lambda row: float(row.get("virtual_time") or 0.0))
    return grouped


def _first_target_row(rows: list[dict[str, Any]], target_accuracy: float | None):
    if target_accuracy is None:
        return None
    for row in rows:
        accuracy = _to_float(row.get("test_accuracy"))
        if accuracy is not None and accuracy >= target_accuracy:
            return row
    return None


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["alpha"],
            row["split"],
            row["num_clients"],
            row["algorithm"],
        )
        grouped[key].append(row)

    out = []
    for key, group in sorted(grouped.items()):
        dataset, alpha, split, num_clients, algorithm = key
        out.append(
            {
                "dataset": dataset,
                "alpha": alpha,
                "split": split,
                "num_clients": num_clients,
                "algorithm": algorithm,
                "num_runs": len(group),
                "final_accuracy_mean": _mean(group, "final_accuracy"),
                "final_accuracy_std": _std(group, "final_accuracy"),
                "final_loss_mean": _mean(group, "final_loss"),
                "time_to_target_mean": _mean(group, "time_to_target"),
                "time_to_target_std": _std(group, "time_to_target"),
                "bytes_to_target_mean": _mean(group, "bytes_to_target"),
                "final_transmitted_bytes_mean": _mean(group, "final_transmitted_bytes"),
            }
        )
    return out


def _mean(rows: list[dict[str, Any]], field: str):
    values = [_to_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else ""


def _std(rows: list[dict[str, Any]], field: str):
    values = [_to_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return stdev(values) if len(values) > 1 else 0.0 if values else ""


def _to_float(value):
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
