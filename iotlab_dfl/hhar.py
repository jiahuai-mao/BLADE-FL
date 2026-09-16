from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import numpy as np


HHAR_LABELS = {
    "bike": 0,
    "sit": 1,
    "stand": 2,
    "walk": 3,
    "stairsup": 4,
    "stairsdown": 5,
}


def load_hhar_stats(data_dir, window_size=128, step_size=64, verbose=True, cache_dir=None):
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")
    root = hhar_root(Path(data_dir))
    paths = sensor_paths(root, "accelerometer")
    if not paths:
        raise FileNotFoundError("HHAR accelerometer CSV files were not found under %s" % root)
    cache = cache_path(root, paths, window_size, step_size, cache_dir)
    if cache.exists():
        if verbose:
            print("[hhar] loading cached features from %s" % cache, flush=True)
        with np.load(cache) as payload:
            return (
                payload["features"].astype(np.float32, copy=False),
                payload["labels"].astype(np.int64, copy=False),
                payload["group_ids"].astype(np.int64, copy=False),
                len(HHAR_LABELS),
            )

    features, labels, group_ids = [], [], []
    group_to_id = {}
    for path in paths:
        before = len(features)
        if verbose:
            print("[hhar] reading %s (%.1f MB)" % (path.name, path.stat().st_size / (1024.0 * 1024.0)), flush=True)
        load_csv(path, window_size, step_size, features, labels, group_ids, group_to_id, verbose)
        if verbose:
            print("[hhar] finished %s: windows=%d" % (path.name, len(features) - before), flush=True)
    if not features:
        raise RuntimeError("HHAR files were found, but no labelled windows could be constructed")
    result = (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
        len(HHAR_LABELS),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name("%s.%d.tmp.npz" % (cache.stem, os.getpid()))
    np.savez(temporary, features=result[0], labels=result[1], group_ids=result[2])
    temporary.replace(cache)
    if verbose:
        print("[hhar] total windows=%d groups=%d" % (len(features), len(group_to_id)), flush=True)
        print("[hhar] cached features at %s" % cache, flush=True)
    return result


def load_csv(path, window_size, step_size, features, labels, group_ids, group_to_id, verbose):
    buffers = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("HHAR CSV has no header: %s" % path)
        columns = {item.lower(): item for item in reader.fieldnames}
        required = {"x", "y", "z", "gt"}
        if not required.issubset(columns):
            raise ValueError("HHAR CSV is missing columns %s: %s" % (sorted(required), path))
        group_columns = [columns[name] for name in ("user", "model", "device") if name in columns]
        for row_index, row in enumerate(reader, 1):
            label_name = row[columns["gt"]].strip().lower().replace("_", "")
            if label_name not in HHAR_LABELS:
                continue
            try:
                point = (float(row[columns["x"]]), float(row[columns["y"]]), float(row[columns["z"]]))
            except (TypeError, ValueError):
                continue
            group_key = "|".join(row[item] for item in group_columns) if group_columns else Path(path).stem
            if group_key not in group_to_id:
                group_to_id[group_key] = len(group_to_id)
            key = (group_key, label_name)
            buffer = buffers.setdefault(key, [])
            buffer.append(point)
            while len(buffer) >= window_size:
                window = np.asarray(buffer[:window_size], dtype=np.float32)
                features.append(window_stats(window))
                labels.append(HHAR_LABELS[label_name])
                group_ids.append(group_to_id[group_key])
                del buffer[:step_size]
            if verbose and row_index % 1000000 == 0:
                print("[hhar] %s: rows=%d windows=%d" % (Path(path).name, row_index, len(features)), flush=True)


def window_stats(window):
    return np.concatenate(
        [window.mean(axis=0), window.std(axis=0), window.min(axis=0), window.max(axis=0)]
    ).astype(np.float32)


def sensor_paths(root, sensor):
    candidates = [
        root / ("Phones_%s.csv" % sensor),
        root / ("Watch_%s.csv" % sensor),
        root / ("phones_%s.csv" % sensor),
        root / ("watch_%s.csv" % sensor),
    ]
    existing = [item for item in candidates if item.exists()]
    if existing:
        return existing
    return [item for item in sorted(root.glob("*.csv")) if sensor in item.name.lower()]


def hhar_root(data_dir):
    for name in ("HHAR", "hhar", "Activity recognition exp"):
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    return data_dir


def cache_path(root, paths, window_size, step_size, cache_dir=None):
    digest = hashlib.sha1()
    digest.update(("window=%d|step=%d|features=stats|sensor=acc" % (window_size, step_size)).encode("ascii"))
    for path in paths:
        stat = path.stat()
        digest.update(("|%s:%d:%d" % (path.name, stat.st_size, stat.st_mtime_ns)).encode("utf-8"))
    base = Path(cache_dir) if cache_dir is not None else root / ".cache"
    return base / ("iotlab_hhar_%s.npz" % digest.hexdigest()[:16])
