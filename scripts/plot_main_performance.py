from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False


METHOD_ORDER = [
    "clustered_async",
    "fedavg",
    "dpsgd",
    "adpsgd",
    "clustered_async_no_mixing",
    "random_clustered_async",
]

METHOD_LABELS = {
    "clustered_async": "Proposed",
    "fedavg": "FedAvg",
    "dpsgd": "D-PSGD",
    "adpsgd": "AD-PSGD",
    "clustered_async_no_mixing": "No inter-cluster mixing",
    "random_clustered_async": "Random clustered async",
}

METHOD_COLORS = {
    "clustered_async": "#0F4D92",
    "fedavg": "#484878",
    "dpsgd": "#7884B4",
    "adpsgd": "#B4C0E4",
    "clustered_async_no_mixing": "#E4CCD8",
    "random_clustered_async": "#A8A8A8",
}

METHOD_MARKERS = {
    "clustered_async": "o",
    "fedavg": "s",
    "dpsgd": "^",
    "adpsgd": "D",
    "clustered_async_no_mixing": "P",
    "random_clustered_async": "X",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Nature-style main performance figures from training results.")
    parser.add_argument("--results-dir", default="results/training")
    parser.add_argument("--output-dir", default="results/figures/main_performance")
    parser.add_argument("--fig1-dataset", default="cifar10")
    parser.add_argument("--fig1-alpha", type=float, default=0.1)
    parser.add_argument("--target-accuracy", type=float, default=0.70)
    parser.add_argument("--formats", default="svg,pdf,tiff,png")
    parser.add_argument("--include-random", action="store_true", help="Include random clustered async if present.")
    args = parser.parse_args()

    runs = read_runs(Path(args.results_dir))
    if not runs:
        raise SystemExit(f"No completed runs with metrics.csv and config.json found under {args.results_dir}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    methods = METHOD_ORDER if args.include_random else [m for m in METHOD_ORDER if m != "random_clustered_async"]

    fig_run = select_figure_run(runs, args.fig1_dataset, args.fig1_alpha)
    if fig_run is None:
        fig_run = select_fallback_run(runs)
        if fig_run is None:
            raise SystemExit("No run has plottable metrics.")
        requested = f"{args.fig1_dataset}, alpha={args.fig1_alpha:g}"
        actual = describe_run(fig_run)
        print(f"[plot] requested Fig.1 run ({requested}) not found; using current available run: {actual}")
    else:
        print(f"[plot] using Fig.1 run: {describe_run(fig_run)}")

    figure_methods = [method for method in methods if method in fig_run["metrics_by_algorithm"]]
    if not figure_methods:
        raise SystemExit(f"No requested algorithms found in {fig_run['run_dir']}.")

    save_source_metrics(output_dir / "source_fig1_metrics.csv", fig_run, figure_methods)
    plot_accuracy_vs_time(fig_run, figure_methods, output_dir / "fig1a_accuracy_vs_time", formats)
    plot_time_to_target(fig_run, figure_methods, args.target_accuracy, output_dir / "fig1b_time_to_target", formats)
    plot_accuracy_vs_bytes(fig_run, figure_methods, output_dir / "fig1c_accuracy_vs_bytes", formats)
    plot_figure1_combined(fig_run, figure_methods, args.target_accuracy, output_dir / "figure1_main_performance", formats)
    plot_loss_supplements(fig_run, figure_methods, output_dir, formats)

    table_rows = build_table_rows(runs, methods)
    write_csv(output_dir / "source_table1_final_accuracy.csv", table_rows)
    plot_table1(table_rows, output_dir / "table1_final_accuracy", formats)

    print(f"[plot] output_dir={output_dir}")


def read_runs(results_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for run_dir in sorted(results_dir.glob("*")):
        metrics_path = run_dir / "metrics.csv"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        metrics = read_metrics(metrics_path)
        if not metrics:
            continue
        runs.append(
            {
                "run_dir": run_dir,
                "config": config,
                "metrics": metrics,
                "metrics_by_algorithm": group_by_algorithm(metrics),
            }
        )
    return runs


def read_metrics(path: Path) -> list[dict[str, Any]]:
    numeric_fields = {
        "step",
        "K",
        "virtual_time",
        "train_loss",
        "test_loss",
        "test_accuracy",
        "best_accuracy",
        "mean_staleness",
        "max_staleness",
        "transmitted_bytes_proxy",
        "model_divergence",
    }
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for field in numeric_fields:
                parsed[field] = to_float(parsed.get(field))
            rows.append(parsed)
    return rows


def group_by_algorithm(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("algorithm", ""))].append(row)
    for algo_rows in grouped.values():
        algo_rows.sort(key=lambda row: row.get("virtual_time") or 0.0)
    return dict(grouped)


def select_figure_run(runs: list[dict[str, Any]], dataset: str, alpha: float) -> dict[str, Any] | None:
    candidates = [
        run for run in runs
        if str(run["config"].get("dataset", "")).lower() == dataset.lower()
        and floats_equal(to_float(run["config"].get("dirichlet_alpha")), alpha)
    ]
    return best_run(candidates)


def select_fallback_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    main = [run for run in runs if str(run["config"].get("dataset", "")).lower() in {"cifar10", "hhar"}]
    return best_run(main) or best_run(runs)


def best_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    return max(runs, key=lambda run: (len(run["metrics_by_algorithm"]), len(run["metrics"])))


def plot_accuracy_vs_time(run: dict[str, Any], methods: list[str], out_base: Path, formats: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    time_scale, time_label = axis_scale(max_value(run, methods, "virtual_time"), "time")
    for method in methods:
        rows = run["metrics_by_algorithm"][method]
        x = [(row["virtual_time"] or 0.0) / time_scale for row in rows]
        y = [(row["test_accuracy"] or math.nan) * 100.0 for row in rows]
        ax.plot(
            x,
            y,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=3.2,
            linewidth=1.4 if method == "clustered_async" else 1.0,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel(time_label)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title_for_run(run))
    add_panel_label(ax, "a")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.65)
    ax.legend(loc="lower right", fontsize=5.8)
    save_figure(fig, out_base, formats)


def plot_time_to_target(run: dict[str, Any], methods: list[str], target: float, out_base: Path, formats: list[str]) -> None:
    values = []
    labels = []
    colors = []
    missing = []
    for method in methods:
        row = first_target_row(run["metrics_by_algorithm"][method], target)
        labels.append(METHOD_LABELS[method])
        colors.append(METHOD_COLORS[method])
        if row is None:
            values.append(0.0)
            missing.append(True)
        else:
            values.append(row["virtual_time"] or 0.0)
            missing.append(False)
    scale, xlabel = axis_scale(max(values) if values else 0.0, "time")
    scaled = [value / scale for value in values]

    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    y_positions = list(range(len(labels)))
    bars = ax.barh(y_positions, scaled, color=colors, edgecolor="#272727", linewidth=0.45)
    for bar, is_missing in zip(bars, missing):
        if is_missing:
            bar.set_facecolor("#F0F0F0")
            bar.set_hatch("///")
    for idx, (value, is_missing) in enumerate(zip(scaled, missing)):
        text = "n.r." if is_missing else f"{value:.1f}"
        xpos = value if value > 0 else max(scaled + [1.0]) * 0.02
        ax.text(xpos, idx, text, va="center", ha="left", fontsize=5.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Time to {target * 100:.0f}% accuracy ({xlabel.split('(')[-1].rstrip(')')})")
    ax.set_title(title_for_run(run))
    add_panel_label(ax, "b")
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.45, alpha=0.65)
    save_figure(fig, out_base, formats)


def plot_accuracy_vs_bytes(run: dict[str, Any], methods: list[str], out_base: Path, formats: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    byte_scale, byte_label = axis_scale(max_value(run, methods, "transmitted_bytes_proxy"), "bytes")
    for method in methods:
        rows = run["metrics_by_algorithm"][method]
        x = [(row["transmitted_bytes_proxy"] or 0.0) / byte_scale for row in rows]
        y = [(row["test_accuracy"] or math.nan) * 100.0 for row in rows]
        ax.plot(
            x,
            y,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=3.2,
            linewidth=1.4 if method == "clustered_async" else 1.0,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel(byte_label)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title_for_run(run))
    add_panel_label(ax, "c")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.65)
    ax.legend(loc="lower right", fontsize=5.8)
    save_figure(fig, out_base, formats)


def plot_figure1_combined(
    run: dict[str, Any],
    methods: list[str],
    target: float,
    out_base: Path,
    formats: list[str],
) -> None:
    fig = plt.figure(figsize=(7.2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.92, 1.15], wspace=0.46)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    time_scale, time_label = axis_scale(max_value(run, methods, "virtual_time"), "time")
    byte_scale, byte_label = axis_scale(max_value(run, methods, "transmitted_bytes_proxy"), "bytes")

    for method in methods:
        rows = run["metrics_by_algorithm"][method]
        y = [(row["test_accuracy"] or math.nan) * 100.0 for row in rows]
        ax_a.plot(
            [(row["virtual_time"] or 0.0) / time_scale for row in rows],
            y,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=2.2,
            linewidth=1.2 if method == "clustered_async" else 0.85,
        )
        ax_c.plot(
            [(row["transmitted_bytes_proxy"] or 0.0) / byte_scale for row in rows],
            y,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=2.2,
            linewidth=1.2 if method == "clustered_async" else 0.85,
        )

    labels = [METHOD_LABELS[method] for method in methods]
    values = []
    missing = []
    for method in methods:
        row = first_target_row(run["metrics_by_algorithm"][method], target)
        if row is None:
            values.append(0.0)
            missing.append(True)
        else:
            values.append(row["virtual_time"] or 0.0)
            missing.append(False)
    bar_scale, _ = axis_scale(max(values) if values else 0.0, "time")
    scaled = [value / bar_scale for value in values]
    bars = ax_b.barh(
        list(range(len(labels))),
        scaled,
        color=[METHOD_COLORS[method] for method in methods],
        edgecolor="#272727",
        linewidth=0.35,
    )
    for idx, (bar, value, is_missing) in enumerate(zip(bars, scaled, missing)):
        if is_missing:
            bar.set_facecolor("#F0F0F0")
            bar.set_hatch("///")
        ax_b.text(
            value if value > 0 else max(scaled + [1.0]) * 0.02,
            idx,
            "n.r." if is_missing else f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=5.2,
        )

    ax_a.set_xlabel(short_axis_label(time_label))
    ax_a.set_ylabel("Test accuracy (%)")
    ax_a.grid(axis="y", color="#D8D8D8", linewidth=0.35, alpha=0.65)
    add_panel_label(ax_a, "a")

    ax_b.set_yticks(list(range(len(labels))))
    ax_b.set_yticklabels(labels, fontsize=5.7)
    ax_b.invert_yaxis()
    ax_b.set_xlabel(f"Time to {target * 100:.0f}% accuracy\n({short_axis_unit(time_label)})")
    ax_b.grid(axis="x", color="#D8D8D8", linewidth=0.35, alpha=0.65)
    add_panel_label(ax_b, "b")

    ax_c.set_xlabel(short_axis_label(byte_label))
    ax_c.set_ylabel("Test accuracy (%)")
    ax_c.grid(axis="y", color="#D8D8D8", linewidth=0.35, alpha=0.65)
    add_panel_label(ax_c, "c")

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=1.1,
            markersize=3.0,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), bbox_to_anchor=(0.55, 1.09), fontsize=5.8)
    fig.suptitle(title_for_run(run), y=1.16, fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.72, wspace=0.46)
    fig._skip_tight_layout = True
    save_figure(fig, out_base, formats)


def plot_loss_supplements(run: dict[str, Any], methods: list[str], output_dir: Path, formats: list[str]) -> None:
    for field, ylabel, name in [
        ("test_loss", "Test loss", "figs1_test_loss_vs_time"),
        ("train_loss", "Train loss", "figs2_train_loss_vs_time"),
        ("test_loss", "Test loss", "figs3_test_loss_vs_bytes"),
        ("train_loss", "Train loss", "figs4_train_loss_vs_bytes"),
    ]:
        x_field = "transmitted_bytes_proxy" if name.endswith("bytes") else "virtual_time"
        kind = "bytes" if x_field == "transmitted_bytes_proxy" else "time"
        scale, xlabel = axis_scale(max_value(run, methods, x_field), kind)
        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        for method in methods:
            rows = [row for row in run["metrics_by_algorithm"][method] if row.get(field) is not None]
            if not rows:
                continue
            x = [(row[x_field] or 0.0) / scale for row in rows]
            y = [row[field] for row in rows]
            ax.plot(x, y, color=METHOD_COLORS[method], linewidth=1.0, marker=METHOD_MARKERS[method], markersize=2.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title_for_run(run))
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.65)
        save_figure(fig, output_dir / name, formats)


def build_table_rows(runs: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    for run in runs:
        config = run["config"]
        dataset = str(config.get("dataset", ""))
        if dataset.lower() not in {"cifar10", "hhar"}:
            continue
        alpha = str(config.get("dirichlet_alpha", ""))
        clients = str(config.get("num_clients", ""))
        for method in methods:
            rows = run["metrics_by_algorithm"].get(method)
            if not rows:
                continue
            final_acc = rows[-1].get("test_accuracy")
            if final_acc is None:
                continue
            key = (dataset, alpha, method)
            grouped[key].append(final_acc * 100.0)
            meta[key] = {
                "dataset": dataset,
                "alpha": alpha,
                "num_clients": clients,
                "algorithm": method,
                "method": METHOD_LABELS[method],
            }
    out = []
    for key in sorted(grouped, key=lambda item: (item[0], float_or_inf(item[1]), METHOD_ORDER.index(item[2]))):
        values = grouped[key]
        row = dict(meta[key])
        row["num_runs"] = len(values)
        row["final_accuracy_mean"] = mean(values)
        row["final_accuracy_std"] = stdev(values) if len(values) > 1 else 0.0
        out.append(row)
    return out


def plot_table1(rows: list[dict[str, Any]], out_base: Path, formats: list[str]) -> None:
    if not rows:
        return
    columns = ["Dataset", "alpha", "Method", "Final accuracy (%)", "n"]
    cell_text = []
    for row in rows:
        acc = f"{row['final_accuracy_mean']:.1f}"
        if row["num_runs"] > 1:
            acc = f"{row['final_accuracy_mean']:.1f} +/- {row['final_accuracy_std']:.1f}"
        cell_text.append([row["dataset"].upper(), row["alpha"], row["method"], acc, str(row["num_runs"])])

    fig_height = max(1.6, 0.24 * len(cell_text) + 0.55)
    fig, ax = plt.subplots(figsize=(6.8, fig_height))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.22)
    for (row_idx, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D8D8D8")
        cell.set_linewidth(0.35)
        if row_idx == 0:
            cell.set_facecolor("#E4E4F0")
            cell.set_text_props(weight="bold")
    ax.set_title("Table 1 | Final accuracy under fixed training budget", loc="left", fontsize=8, fontweight="bold")
    save_figure(fig, out_base, formats)


def save_source_metrics(path: Path, run: dict[str, Any], methods: list[str]) -> None:
    fields = [
        "dataset",
        "alpha",
        "num_clients",
        "algorithm",
        "method",
        "step",
        "virtual_time",
        "train_loss",
        "test_loss",
        "test_accuracy",
        "best_accuracy",
        "transmitted_bytes_proxy",
    ]
    rows = []
    config = run["config"]
    for method in methods:
        for row in run["metrics_by_algorithm"][method]:
            rows.append(
                {
                    "dataset": config.get("dataset", ""),
                    "alpha": config.get("dirichlet_alpha", ""),
                    "num_clients": config.get("num_clients", ""),
                    "algorithm": method,
                    "method": METHOD_LABELS[method],
                    "step": row.get("step"),
                    "virtual_time": row.get("virtual_time"),
                    "train_loss": row.get("train_loss"),
                    "test_loss": row.get("test_loss"),
                    "test_accuracy": row.get("test_accuracy"),
                    "best_accuracy": row.get("best_accuracy"),
                    "transmitted_bytes_proxy": row.get("transmitted_bytes_proxy"),
                }
            )
    write_csv(path, rows)


def first_target_row(rows: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    for row in rows:
        accuracy = row.get("test_accuracy")
        if accuracy is not None and accuracy >= target:
            return row
    return None


def axis_scale(max_raw: float, kind: str) -> tuple[float, str]:
    if kind == "bytes":
        if max_raw >= 1e9:
            return 1e9, "Transmitted bytes proxy (GB)"
        if max_raw >= 1e6:
            return 1e6, "Transmitted bytes proxy (MB)"
        if max_raw >= 1e3:
            return 1e3, "Transmitted bytes proxy (KB)"
        return 1.0, "Transmitted bytes proxy (bytes)"
    if max_raw >= 1e6:
        return 1e6, "Simulated wall-clock time (x10^6 a.u.)"
    if max_raw >= 1e3:
        return 1e3, "Simulated wall-clock time (x10^3 a.u.)"
    return 1.0, "Simulated wall-clock time (a.u.)"


def short_axis_label(label: str) -> str:
    return label.replace("Simulated wall-clock time", "Time").replace("Transmitted bytes proxy", "Bytes")


def short_axis_unit(label: str) -> str:
    if "(" not in label:
        return "a.u."
    return label.split("(", 1)[1].rstrip(")")


def max_value(run: dict[str, Any], methods: list[str], field: str) -> float:
    values = []
    for method in methods:
        values.extend(row.get(field) or 0.0 for row in run["metrics_by_algorithm"][method])
    return max(values) if values else 0.0


def title_for_run(run: dict[str, Any]) -> str:
    config = run["config"]
    dataset = str(config.get("dataset", "")).upper()
    alpha = config.get("dirichlet_alpha", "")
    clients = config.get("num_clients", "")
    seed = config.get("seed", "")
    return f"{dataset}, Dirichlet alpha={alpha}, N={clients}, seed={seed}"


def describe_run(run: dict[str, Any]) -> str:
    config = run["config"]
    return (
        f"{config.get('dataset')} alpha={config.get('dirichlet_alpha')} "
        f"N={config.get('num_clients')} seed={config.get('seed')} ({run['run_dir']})"
    )


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def save_figure(fig, out_base: Path, formats: list[str]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    if not getattr(fig, "_skip_tight_layout", False):
        fig.tight_layout(pad=1.2)
    for fmt in formats:
        kwargs = {"bbox_inches": "tight"}
        if fmt in {"tiff", "tif", "png"}:
            kwargs["dpi"] = 600
        fig.savefig(out_base.with_suffix(f".{fmt}"), **kwargs)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def floats_equal(left: float | None, right: float) -> bool:
    return left is not None and abs(left - right) < 1e-12


def float_or_inf(value: Any) -> float:
    parsed = to_float(value)
    return parsed if parsed is not None else math.inf


if __name__ == "__main__":
    main()
