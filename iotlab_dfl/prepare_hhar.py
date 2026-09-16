from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from iotlab_dfl.hhar import HHAR_LABELS, load_hhar_stats
from iotlab_dfl.model import initialize_model, parameter_count
from iotlab_dfl.storage import write_json


def main():
    parser = argparse.ArgumentParser(description="Prepare lightweight HHAR shards for FIT IoT-LAB A8 nodes.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="iotlab_dfl/bundle")
    parser.add_argument("--num-clients", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--partition-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--hidden-dims", default=None, help="Comma-separated hidden widths, for example 64,32.")
    parser.add_argument("--feature-set", choices=["legacy12", "rich48"], default="legacy12")
    parser.add_argument("--raw-cache", default=None, help="Six-channel HHAR raw-window NPZ used by rich48.")
    parser.add_argument("--partition", choices=["balanced-groups", "block-dirichlet"], default="balanced-groups")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--block-size", type=int, default=256, help="Maximum windows per temporal block.")
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--min-labels", type=int, default=2)
    parser.add_argument("--min-label-samples", type=int, default=64)
    parser.add_argument("--max-sample-factor", type=float, default=2.5)
    args = parser.parse_args()
    if args.num_clients < 2:
        raise ValueError("num-clients must be at least two")
    output = Path(args.output_dir)
    shard_dir = output / "data"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_set == "rich48":
        if not args.raw_cache:
            raise ValueError("--raw-cache is required for feature-set rich48")
        features, labels, group_ids, num_classes = load_rich48(Path(args.raw_cache))
        sensor_mode = "acc_gyro"
        feature_names = rich48_feature_names()
    else:
        features, labels, group_ids, num_classes = load_hhar_stats(
            Path(args.data_dir),
            window_size=args.window_size,
            step_size=args.step_size,
            verbose=True,
            cache_dir=output / ".cache",
        )
        sensor_mode = "acc"
        feature_names = ["%s_%s" % (stat, axis) for stat in ("mean", "std", "min", "max") for axis in ("x", "y", "z")]
    hidden_dims = parse_hidden_dims(args.hidden_dims, args.hidden_dim)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    partition_seed = args.seed if args.partition_seed is None else args.partition_seed
    model_seed = args.seed if args.model_seed is None else args.model_seed
    train_mask, test_mask = group_split(group_ids, args.test_fraction, split_seed)
    train_x = np.asarray(features[train_mask], dtype=np.float32)
    train_y = np.asarray(labels[train_mask], dtype=np.int64)
    train_groups = np.asarray(group_ids[train_mask], dtype=np.int64)
    train_source_indices = np.flatnonzero(train_mask).astype(np.int64)
    test_x = np.asarray(features[test_mask], dtype=np.float32)
    test_y = np.asarray(labels[test_mask], dtype=np.int64)

    mean = train_x.mean(axis=0).astype(np.float32)
    std = np.maximum(train_x.std(axis=0), np.float32(1e-6)).astype(np.float32)
    train_x = ((train_x - mean) / std).astype(np.float32)
    test_x = ((test_x - mean) / std).astype(np.float32)
    if args.partition == "block-dirichlet":
        min_samples = args.min_samples
        if min_samples is None:
            min_samples = max(1, int(len(train_y) / float(args.num_clients) * 0.28))
        assignments, blocks, partition_stats = block_dirichlet_partition(
            train_y,
            train_groups,
            train_source_indices,
            args.num_clients,
            num_classes,
            args.dirichlet_alpha,
            args.block_size,
            min_samples,
            args.min_labels,
            args.min_label_samples,
            args.max_sample_factor,
            partition_seed,
        )
    else:
        group_assignments = merge_groups(train_groups, args.num_clients, args.seed)
        assignments = [
            np.flatnonzero(np.isin(train_groups, np.asarray(sorted(group_set), dtype=train_groups.dtype)))
            for group_set in group_assignments
        ]
        blocks = None
        partition_stats = None

    clients = []
    for client_id, selected in enumerate(assignments):
        selected = np.asarray(selected, dtype=np.int64)
        client_x = train_x[selected]
        client_y = train_y[selected]
        if len(client_y) == 0:
            raise RuntimeError("client %d received no HHAR windows" % client_id)
        counts = np.bincount(client_y, minlength=num_classes).astype(np.int64)
        path = shard_dir / ("client_%02d.npz" % client_id)
        np.savez_compressed(path, x=client_x, y=client_y)
        clients.append(
            {
                "client_id": client_id,
                "file": "data/%s" % path.name,
                "num_samples": int(len(client_y)),
                "num_groups": int(len(np.unique(train_groups[selected]))),
                "group_ids": sorted(int(item) for item in np.unique(train_groups[selected])),
                "label_counts": counts.tolist(),
                "label_entropy": label_entropy(counts),
                "js_divergence": js_divergence(counts, np.bincount(train_y, minlength=num_classes)),
                "sha256": sha256(path),
            }
        )
        if blocks is not None:
            clients[-1]["block_ids"] = [int(item) for item in partition_stats["client_block_ids"][client_id]]

    np.savez_compressed(shard_dir / "test.npz", x=test_x, y=test_y)
    initial = initialize_model(train_x.shape[1], hidden_dims, num_classes, model_seed)
    np.save(output / "initial_model.npy", initial)
    np.savez(output / "normalization.npz", mean=mean, std=std)
    manifest = {
        "dataset": "hhar",
        "feature_mode": args.feature_set,
        "feature_names": feature_names,
        "sensor_mode": sensor_mode,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "split": "group-disjoint-test/block-dirichlet-train" if blocks is not None else "group",
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "split_seed": split_seed,
        "partition_seed": partition_seed,
        "model_seed": model_seed,
        "num_clients": args.num_clients,
        "num_classes": num_classes,
        "class_names": [name for name, _ in sorted(HHAR_LABELS.items(), key=lambda item: item[1])],
        "input_dim": int(train_x.shape[1]),
        "hidden_dim": hidden_dims[0],
        "hidden_dims": hidden_dims,
        "parameter_count": parameter_count(train_x.shape[1], hidden_dims, num_classes),
        "model_bytes": int(initial.nbytes),
        "train_samples": int(len(train_y)),
        "test_samples": int(len(test_y)),
        "train_group_ids": sorted(int(item) for item in np.unique(train_groups)),
        "test_group_ids": sorted(int(item) for item in np.unique(group_ids[test_mask])),
        "clients": clients,
        "test_sha256": sha256(shard_dir / "test.npz"),
        "initial_model_sha256": sha256(output / "initial_model.npy"),
    }
    if blocks is not None:
        block_manifest = []
        owner_by_block = {}
        for client_id, block_ids in enumerate(partition_stats["client_block_ids"]):
            for block_id in block_ids:
                owner_by_block[int(block_id)] = client_id
        for block in blocks:
            block_manifest.append(
                {
                    "block_id": int(block["block_id"]),
                    "client_id": int(owner_by_block[int(block["block_id"])]),
                    "group_id": int(block["group_id"]),
                    "label": int(block["label"]),
                    "num_samples": int(len(block["indices"])),
                    "source_index_start": int(block["source_indices"][0]),
                    "source_index_end": int(block["source_indices"][-1]),
                }
            )
        partition = {
            "method": "label-wise-dirichlet-guided-temporal-blocks",
            "partition_seed": partition_seed,
            "dirichlet_alpha": args.dirichlet_alpha,
            "maximum_block_windows": args.block_size,
            "minimum_client_samples": partition_stats["minimum_client_samples"],
            "minimum_client_labels": args.min_labels,
            "minimum_samples_per_required_label": args.min_label_samples,
            "maximum_client_samples_soft": partition_stats["maximum_client_samples_soft"],
            "maximum_sample_factor": args.max_sample_factor,
            "random_window_mixing": False,
            "all_training_windows_assigned_once": True,
            "blocks": block_manifest,
            "client_sample_min": min(item["num_samples"] for item in clients),
            "client_sample_max": max(item["num_samples"] for item in clients),
            "client_js_divergence_mean": float(np.mean([item["js_divergence"] for item in clients])),
            "client_js_divergence_max": max(item["js_divergence"] for item in clients),
        }
        partition["assignment_sha256"] = hashlib.sha256(
            json.dumps(block_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest["partition"] = partition
    if args.raw_cache:
        manifest["raw_cache_source"] = str(Path(args.raw_cache))
        manifest["raw_cache_sha256"] = sha256(Path(args.raw_cache))
    write_json(output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ["num_clients", "train_samples", "test_samples", "parameter_count", "model_bytes"]}, indent=2))


def parse_hidden_dims(value, fallback):
    dims = [int(item.strip()) for item in value.split(",") if item.strip()] if value else [int(fallback)]
    if not dims or any(item <= 0 for item in dims):
        raise ValueError("hidden dimensions must be positive")
    return dims


def load_rich48(path, chunk_size=4096):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("HHAR raw-window cache not found: %s" % path)
    print("[hhar] loading six-channel windows from %s" % path, flush=True)
    with np.load(path) as payload:
        raw = payload["features"]
        labels = payload["labels"].astype(np.int64, copy=False)
        group_ids = payload["group_ids"].astype(np.int64, copy=False)
        num_classes = int(payload["num_classes"]) if "num_classes" in payload else len(HHAR_LABELS)
        if raw.ndim != 3 or raw.shape[1] != 6:
            raise ValueError("rich48 requires raw windows shaped [N, 6, T], got %s" % (raw.shape,))
        features = np.empty((raw.shape[0], 48), dtype=np.float32)
        for begin in range(0, raw.shape[0], chunk_size):
            end = min(begin + chunk_size, raw.shape[0])
            features[begin:end] = rich48_features(np.asarray(raw[begin:end], dtype=np.float32))
            if begin == 0 or end == raw.shape[0] or (begin // chunk_size) % 10 == 0:
                print("[hhar] rich48 windows=%d/%d" % (end, raw.shape[0]), flush=True)
    return features, labels, group_ids, num_classes


def rich48_features(windows):
    axis_parts = [
        windows.mean(axis=2),
        windows.std(axis=2),
        windows.min(axis=2),
        windows.max(axis=2),
        np.sqrt(np.mean(windows * windows, axis=2)),
        np.mean(np.abs(np.diff(windows, axis=2)), axis=2),
    ]
    magnitude_parts = []
    for sensor in (windows[:, :3], windows[:, 3:]):
        magnitude = np.sqrt(np.sum(sensor * sensor, axis=1))
        magnitude_parts.extend(
            [
                magnitude.mean(axis=1, keepdims=True),
                magnitude.std(axis=1, keepdims=True),
                magnitude.min(axis=1, keepdims=True),
                magnitude.max(axis=1, keepdims=True),
                np.sqrt(np.mean(magnitude * magnitude, axis=1, keepdims=True)),
                np.mean(np.abs(np.diff(magnitude, axis=1)), axis=1, keepdims=True),
            ]
        )
    return np.concatenate(axis_parts + magnitude_parts, axis=1).astype(np.float32, copy=False)


def rich48_feature_names():
    names = []
    for stat in ("mean", "std", "min", "max", "rms", "mean_abs_diff"):
        for channel in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"):
            names.append("%s_%s" % (stat, channel))
    for sensor in ("acc_magnitude", "gyro_magnitude"):
        for stat in ("mean", "std", "min", "max", "rms", "mean_abs_diff"):
            names.append("%s_%s" % (stat, sensor))
    return names


def group_split(group_ids, test_fraction, seed):
    unique = np.unique(group_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    target = max(1, int(round(len(group_ids) * test_fraction)))
    selected = []
    count = 0
    for group_id in unique:
        selected.append(int(group_id))
        count += int(np.sum(group_ids == group_id))
        if count >= target:
            break
    test = np.isin(group_ids, np.asarray(selected, dtype=group_ids.dtype))
    if test.all() or not test.any():
        raise RuntimeError("invalid group split")
    return ~test, test


def merge_groups(group_ids, num_clients, seed):
    groups = [(int(group_id), int(np.sum(group_ids == group_id))) for group_id in np.unique(group_ids)]
    if len(groups) < num_clients:
        raise ValueError("HHAR has %d training groups, fewer than %d clients" % (len(groups), num_clients))
    rng = np.random.RandomState(seed)
    tie_breakers = {group_id: float(rng.rand()) for group_id, _ in groups}
    groups.sort(key=lambda item: (-item[1], tie_breakers[item[0]], item[0]))
    assignments = [set() for _ in range(num_clients)]
    loads = [0 for _ in range(num_clients)]
    for group_id, size in groups:
        client_id = min(range(num_clients), key=lambda item: (loads[item], item))
        assignments[client_id].add(group_id)
        loads[client_id] += size
    return assignments


def temporal_blocks(labels, group_ids, source_indices, maximum_size):
    if maximum_size <= 0:
        raise ValueError("block-size must be positive")
    labels = np.asarray(labels, dtype=np.int64)
    group_ids = np.asarray(group_ids, dtype=np.int64)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if not (len(labels) == len(group_ids) == len(source_indices)):
        raise ValueError("labels, groups and source indices must have equal length")
    changes = np.r_[True, (labels[1:] != labels[:-1]) | (group_ids[1:] != group_ids[:-1])]
    starts = np.flatnonzero(changes)
    ends = np.r_[starts[1:], len(labels)]
    blocks = []
    for run_start, run_end in zip(starts, ends):
        run_size = int(run_end - run_start)
        parts = int(np.ceil(run_size / float(maximum_size)))
        for selected in np.array_split(np.arange(run_start, run_end, dtype=np.int64), parts):
            block_id = len(blocks)
            blocks.append(
                {
                    "block_id": block_id,
                    "group_id": int(group_ids[selected[0]]),
                    "label": int(labels[selected[0]]),
                    "indices": selected,
                    "source_indices": source_indices[selected],
                }
            )
    return blocks


def block_dirichlet_partition(
    labels,
    group_ids,
    source_indices,
    num_clients,
    num_classes,
    alpha,
    block_size,
    min_samples,
    min_labels,
    min_label_samples,
    max_sample_factor,
    seed,
):
    if alpha <= 0:
        raise ValueError("dirichlet-alpha must be positive")
    if min_samples <= 0 or min_labels <= 0 or min_label_samples <= 0:
        raise ValueError("minimum samples and labels must be positive")
    if min_labels > num_classes:
        raise ValueError("minimum labels exceeds number of classes")
    blocks = temporal_blocks(labels, group_ids, source_indices, block_size)
    rng = np.random.RandomState(seed)
    preferences = np.vstack(
        [rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64)) for _ in range(num_classes)]
    )
    label_totals = np.bincount(labels, minlength=num_classes).astype(np.float64)
    targets = preferences * label_totals[:, None]
    loads = np.zeros(num_clients, dtype=np.int64)
    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    client_blocks = [[] for _ in range(num_clients)]
    available = {label: [] for label in range(num_classes)}
    for block in blocks:
        available[int(block["label"])].append(int(block["block_id"]))
    for label in available:
        rng.shuffle(available[label])
        available[label].sort(key=lambda item: len(blocks[item]["indices"]))

    upper = int(np.ceil(max_sample_factor * len(labels) / float(num_clients)))

    def assign(client_id, block_id):
        block = blocks[block_id]
        label = int(block["label"])
        size = len(block["indices"])
        client_blocks[client_id].append(block_id)
        loads[client_id] += size
        counts[client_id, label] += size
        available[label].remove(block_id)

    # Seed every client with its preferred distinct labels before filling sample floors.
    order = list(range(num_clients))
    rng.shuffle(order)
    for client_id in order:
        label_order = sorted(range(num_classes), key=lambda label: (-preferences[label, client_id], label))
        for label in label_order[:min_labels]:
            while counts[client_id, label] < min_label_samples:
                if not available[label]:
                    raise RuntimeError("not enough temporal blocks to seed every client label")
                needed = int(min_label_samples - counts[client_id, label])
                block_id = min(
                    available[label],
                    key=lambda item: (abs(len(blocks[item]["indices"]) - needed), item),
                )
                assign(client_id, block_id)

    while np.any(loads < min_samples):
        client_id = min(range(num_clients), key=lambda item: (loads[item] / float(min_samples), item))
        candidates = []
        for label, block_ids in available.items():
            for block_id in block_ids:
                size = len(blocks[block_id]["indices"])
                overshoot = max(0, int(loads[client_id] + size - min_samples))
                score = preferences[label, client_id] / float(1 + overshoot)
                candidates.append((score, -overshoot, -size, -block_id, block_id))
        if not candidates:
            raise RuntimeError("temporal blocks exhausted before minimum client size was reached")
        assign(client_id, max(candidates)[-1])

    remaining = [block_id for block_ids in available.values() for block_id in block_ids]
    rng.shuffle(remaining)
    remaining.sort(key=lambda item: len(blocks[item]["indices"]), reverse=True)
    for block_id in remaining:
        block = blocks[block_id]
        label = int(block["label"])
        size = len(block["indices"])
        eligible = [client_id for client_id in range(num_clients) if loads[client_id] + size <= upper]
        if not eligible:
            eligible = list(range(num_clients))
        client_id = max(
            eligible,
            key=lambda item: (
                targets[label, item] - counts[item, label],
                preferences[label, item],
                -loads[item],
                -item,
            ),
        )
        assign(client_id, block_id)

    assignments = []
    seen = []
    for block_ids in client_blocks:
        selected = np.concatenate([blocks[item]["indices"] for item in block_ids])
        selected.sort()
        assignments.append(selected)
        seen.extend(selected.tolist())
    if sorted(seen) != list(range(len(labels))):
        raise RuntimeError("partition did not assign every training window exactly once")
    coverage = np.sum(counts >= min_label_samples, axis=1)
    if int(loads.min()) < min_samples or int(coverage.min()) < min_labels:
        raise RuntimeError("partition constraints were not satisfied")
    return assignments, blocks, {
        "client_block_ids": client_blocks,
        "minimum_client_samples": int(min_samples),
        "maximum_client_samples_soft": int(upper),
    }


def label_entropy(counts):
    counts = np.asarray(counts, dtype=np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def js_divergence(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= left.sum()
    right /= right.sum()
    midpoint = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    return float(
        0.5 * np.sum(left[left_mask] * np.log(left[left_mask] / midpoint[left_mask]))
        + 0.5 * np.sum(right[right_mask] * np.log(right[right_mask] / midpoint[right_mask]))
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
