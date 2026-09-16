from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

from iotlab_dfl.storage import read_json, read_jsonl, write_csv


def main():
    parser = argparse.ArgumentParser(description="Build source data from FIT IoT-LAB node logs.")
    parser.add_argument("--results-dir", default="iotlab_dfl/results")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    root = Path(args.results_dir)
    output = Path(args.output_dir) if args.output_dir else root / "source_data"
    output.mkdir(parents=True, exist_ok=True)
    evaluations, updates, messages, client_times, summaries = collect(root)
    write_csv(output / "evaluations.csv", evaluations)
    write_csv(output / "updates.csv", updates)
    write_csv(output / "messages.csv", messages)
    write_csv(output / "client_training_times.csv", client_times)
    write_csv(output / "run_summary.csv", summaries)
    print("[collect] runs=%d evaluations=%d updates=%d messages=%d" % (len(summaries), len(evaluations), len(updates), len(messages)))


def collect(root):
    evaluations, updates, messages, client_times, summaries = [], [], [], [], []
    for run_path in sorted(root.glob("*/run.json")):
        run_dir = run_path.parent
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        run = read_json(run_path)
        summary = read_json(summary_path)
        prefix = {"run_id": run["run_id"], "algorithm": run["algorithm"], "seed": int(run["seed"])}
        for row in read_csv_rows(run_dir / "evaluations.csv"):
            evaluations.append({**prefix, **row})
        node_rows = []
        for log_path in sorted((run_dir / "node_logs").glob("*.jsonl")):
            rows = read_jsonl(log_path)
            node_rows.extend(rows)
            for row in rows:
                if row.get("kind") == "update":
                    updates.append({**prefix, "physical_node": log_path.stem, **row})
                elif row.get("kind") == "message":
                    messages.append({**prefix, "physical_node": log_path.stem, "included_in_training_metric": True, **row})
        controller_rows = read_jsonl(run_dir / "controller_messages.jsonl")
        for row in controller_rows:
            messages.append(
                {
                    **prefix,
                    "physical_node": "controller",
                    "message_type": row.get("operation"),
                    "included_in_training_metric": row.get("category") == "algorithm_setup",
                    **row,
                }
            )
        per_client = {}
        for row in node_rows:
            if row.get("kind") == "update":
                per_client.setdefault(int(row["client_id"]), []).append(float(row["local_train_sec"]))
        for client_id, values in sorted(per_client.items()):
            client_times.append(
                {
                    **prefix,
                    "client_id": client_id,
                    "median_local_train_sec": median(values),
                    "mean_local_train_sec": sum(values) / len(values),
                    "num_updates": len(values),
                }
            )
        idle_fraction, state_totals = compute_idle_fraction(node_rows)
        peer_rows = [row for row in node_rows if row.get("kind") == "message"]
        setup_rows = [row for row in controller_rows if row.get("category") == "algorithm_setup"]
        tx_total = sum(int(row.get("request_bytes", 0)) for row in peer_rows + setup_rows)
        rx_total = sum(int(row.get("response_bytes", 0)) for row in peer_rows + setup_rows)
        message_total = tx_total + rx_total
        summaries.append(
            {
                **prefix,
                **summary,
                "protocol_idle_fraction": idle_fraction,
                "measured_training_tx_bytes": tx_total,
                "measured_training_rx_bytes": rx_total,
                "logged_training_bytes": message_total,
                **{"state_%s_sec" % key.lower(): value for key, value in state_totals.items()},
            }
        )
    return evaluations, updates, messages, client_times, summaries


def compute_idle_fraction(rows):
    by_client = {}
    for row in rows:
        if row.get("kind") != "interval" or row.get("state") not in {"LOCAL_TRAIN", "COMMUNICATION", "AGGREGATION"}:
            continue
        by_client.setdefault(int(row["client_id"]), []).append((float(row["start"]), float(row["end"]), str(row["state"])))
    idle_total = 0.0
    window_total = 0.0
    state_totals = {"LOCAL_TRAIN": 0.0, "COMMUNICATION": 0.0, "AGGREGATION": 0.0, "IDLE_WAIT": 0.0}
    for intervals in by_client.values():
        if not intervals:
            continue
        start = min(item[0] for item in intervals)
        end = max(item[1] for item in intervals)
        window = max(0.0, end - start)
        busy = union_duration([(item[0], item[1]) for item in intervals])
        idle = max(0.0, window - busy)
        idle_total += idle
        window_total += window
        state_totals["IDLE_WAIT"] += idle
        for state in ["LOCAL_TRAIN", "COMMUNICATION", "AGGREGATION"]:
            state_totals[state] += union_duration([(left, right) for left, right, label in intervals if label == state])
    return (idle_total / window_total if window_total > 0 else 0.0), state_totals


def union_duration(intervals):
    if not intervals:
        return 0.0
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(max(0.0, end - start) for start, end in merged)


def read_csv_rows(path):
    import csv

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
