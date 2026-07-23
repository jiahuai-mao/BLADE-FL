from __future__ import annotations

import csv
import gzip
import pickle
import struct
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from .base import ClientState
from clustered_dfl.types import ClientMetadata


def load_torch_datasets(dataset: str, data_dir: str | Path, seed: int = 0):
    dataset = dataset.lower()
    data_dir = Path(data_dir)
    if dataset == "synthetic":
        return _synthetic_dataset(seed)
    if dataset == "mnist":
        return _mnist_dataset(data_dir)
    if dataset == "cifar10":
        return _cifar10_dataset(data_dir)
    if dataset == "cifar100":
        return _cifar100_dataset(data_dir)
    if dataset == "hhar":
        return _hhar_dataset(data_dir, seed)
    if dataset in {"fashionmnist", "svhn"}:
        return _torchvision_dataset(dataset, data_dir)
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_client_states(
    train_dataset,
    client_indices: Sequence[Sequence[int]],
    metadata: Sequence[ClientMetadata],
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> list[ClientState]:
    metadata_by_id = {item.client_id: item for item in metadata}
    states = []
    for cid, indices in enumerate(client_indices):
        generator = torch.Generator()
        generator.manual_seed(seed + cid)
        loader = DataLoader(
            Subset(train_dataset, list(indices)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
        states.append(
            ClientState(
                client_id=cid,
                train_loader=loader,
                metadata=metadata_by_id[cid],
            )
        )
    return states


def make_test_loader(test_dataset, batch_size: int, num_workers: int = 0) -> DataLoader:
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _synthetic_dataset(seed: int):
    rng = np.random.default_rng(seed)
    num_classes = 10
    input_dim = 20
    centers = rng.normal(0.0, 2.0, size=(num_classes, input_dim)).astype(np.float32)

    def build(repeats: int):
        labels = np.tile(np.arange(num_classes, dtype=np.int64), repeats)
        features = centers[labels] + rng.normal(0.0, 0.8, size=(len(labels), input_dim)).astype(np.float32)
        return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))

    return build(100), build(20), num_classes


def _mnist_dataset(data_dir: Path):
    root = data_dir / "MNIST" / "raw"
    train_x = _read_mnist_images(root / "train-images-idx3-ubyte", root / "train-images-idx3-ubyte.gz")
    train_y = _read_mnist_labels(root / "train-labels-idx1-ubyte", root / "train-labels-idx1-ubyte.gz")
    test_x = _read_mnist_images(root / "t10k-images-idx3-ubyte", root / "t10k-images-idx3-ubyte.gz")
    test_y = _read_mnist_labels(root / "t10k-labels-idx1-ubyte", root / "t10k-labels-idx1-ubyte.gz")
    return TensorDataset(train_x, train_y), TensorDataset(test_x, test_y), 10


def _cifar10_dataset(data_dir: Path):
    root = data_dir / "cifar-10-batches-py"
    train_images = []
    train_labels = []
    for idx in range(1, 6):
        path = root / f"data_batch_{idx}"
        if not path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10 batch: {path}")
        with path.open("rb") as f:
            payload = pickle.load(f, encoding="latin1")
        train_images.append(payload["data"])
        train_labels.extend(payload["labels"])
    test_path = root / "test_batch"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing CIFAR-10 test batch: {test_path}")
    with test_path.open("rb") as f:
        test_payload = pickle.load(f, encoding="latin1")
    train_x = _cifar_tensor(np.concatenate(train_images, axis=0))
    test_x = _cifar_tensor(test_payload["data"])
    return (
        TensorDataset(train_x, torch.tensor(train_labels, dtype=torch.long)),
        TensorDataset(test_x, torch.tensor(test_payload["labels"], dtype=torch.long)),
        10,
    )


def _cifar100_dataset(data_dir: Path):
    root = data_dir / "cifar-100-python"
    train_path = root / "train"
    test_path = root / "test"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing CIFAR-100 files under {root}")
    with train_path.open("rb") as f:
        train_payload = pickle.load(f, encoding="latin1")
    with test_path.open("rb") as f:
        test_payload = pickle.load(f, encoding="latin1")
    return (
        TensorDataset(_cifar_tensor(train_payload["data"]), torch.tensor(train_payload["fine_labels"], dtype=torch.long)),
        TensorDataset(_cifar_tensor(test_payload["data"]), torch.tensor(test_payload["fine_labels"], dtype=torch.long)),
        100,
    )


def _torchvision_dataset(dataset: str, data_dir: Path):
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(f"torchvision is required to load {dataset}.") from exc
    transform = transforms.ToTensor()
    if dataset == "fashionmnist":
        return (
            datasets.FashionMNIST(root=str(data_dir), train=True, download=False, transform=transform),
            datasets.FashionMNIST(root=str(data_dir), train=False, download=False, transform=transform),
            10,
        )
    if dataset == "svhn":
        return (
            datasets.SVHN(root=str(data_dir), split="train", download=False, transform=transform),
            datasets.SVHN(root=str(data_dir), split="test", download=False, transform=transform),
            10,
        )
    raise ValueError(f"Unsupported torchvision dataset: {dataset}")


def _read_mnist_images(raw_path: Path, gzip_path: Path) -> torch.Tensor:
    path = raw_path if raw_path.exists() else gzip_path
    if not path.exists():
        raise FileNotFoundError(f"Missing MNIST images: {raw_path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file magic: {magic}")
        data = np.frombuffer(f.read(size * rows * cols), dtype=np.uint8)
    data = data.reshape(size, 1, rows, cols).astype(np.float32) / 255.0
    return torch.from_numpy(data)


def _read_mnist_labels(raw_path: Path, gzip_path: Path) -> torch.Tensor:
    path = raw_path if raw_path.exists() else gzip_path
    if not path.exists():
        raise FileNotFoundError(f"Missing MNIST labels: {raw_path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file magic: {magic}")
        labels = np.frombuffer(f.read(size), dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(labels)


def _cifar_tensor(flat_data) -> torch.Tensor:
    arr = np.asarray(flat_data, dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0
    return torch.from_numpy(arr)


HHAR_LABELS = {
    "bike": 0,
    "sit": 1,
    "stand": 2,
    "walk": 3,
    "stairsup": 4,
    "stairsdown": 5,
}


def load_hhar_arrays(
    data_dir: Path,
    window_size: int = 128,
    step_size: int = 128,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    root = _hhar_root(data_dir)
    candidates = [
        root / "Phones_accelerometer.csv",
        root / "Watch_accelerometer.csv",
        root / "phones_accelerometer.csv",
        root / "watch_accelerometer.csv",
    ]
    paths = [path for path in candidates if path.exists()]
    if not paths:
        csv_paths = sorted(root.glob("*.csv"))
        paths = [path for path in csv_paths if "accelerometer" in path.name.lower()]
    if not paths:
        raise FileNotFoundError(
            "HHAR accelerometer CSV files not found. Expected data/HHAR/Phones_accelerometer.csv "
            "or data/hhar/Watch_accelerometer.csv."
        )

    features = []
    labels = []
    group_ids = []
    group_to_id: dict[str, int] = {}
    if verbose:
        file_names = ", ".join(path.name for path in paths)
        print(f"[hhar] loading accelerometer CSV files from {root}: {file_names}", flush=True)
    for path in paths:
        before_windows = len(features)
        if verbose:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"[hhar] reading {path.name} ({size_mb:.1f} MB)...", flush=True)
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is None:
            _load_hhar_csv_streaming(
                path,
                window_size,
                step_size,
                features,
                labels,
                group_ids,
                group_to_id,
                verbose,
            )
            if verbose:
                print(f"[hhar] finished {path.name}: windows={len(features) - before_windows}", flush=True)
            continue
        frame = pd.read_csv(path)
        columns = {column.lower(): column for column in frame.columns}
        required = {"x", "y", "z", "gt"}
        if not required.issubset(columns):
            raise ValueError(f"HHAR CSV missing columns {sorted(required)}: {path}")
        user_col = columns.get("user")
        model_col = columns.get("model")
        device_col = columns.get("device")
        group_cols = [col for col in [user_col, model_col, device_col] if col is not None]
        if not group_cols:
            frame["_group"] = path.stem
            group_cols = ["_group"]
        frame = frame.dropna(subset=[columns["x"], columns["y"], columns["z"], columns["gt"]])
        frame["_label"] = frame[columns["gt"]].astype(str).str.lower().str.replace("_", "", regex=False)
        frame = frame[frame["_label"].isin(HHAR_LABELS)]
        if frame.empty:
            continue
        for group_values, group_frame in frame.groupby(group_cols + ["_label"], sort=True):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            label_name = str(group_values[-1])
            group_key = "|".join(str(value) for value in group_values[:-1])
            if group_key not in group_to_id:
                group_to_id[group_key] = len(group_to_id)
            values = group_frame[[columns["x"], columns["y"], columns["z"]]].to_numpy(dtype=np.float32)
            for start in range(0, max(len(values) - window_size + 1, 0), step_size):
                window = values[start : start + window_size]
                features.append(_hhar_window_features(window))
                labels.append(HHAR_LABELS[label_name])
                group_ids.append(group_to_id[group_key])
        if verbose:
            print(f"[hhar] finished {path.name}: windows={len(features) - before_windows}", flush=True)
    if not features:
        raise RuntimeError("HHAR CSV files were found, but no labelled windows could be constructed.")
    if verbose:
        print(f"[hhar] total windows={len(features)} groups={len(group_to_id)}", flush=True)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
        len(HHAR_LABELS),
    )


def _load_hhar_csv_streaming(
    path: Path,
    window_size: int,
    step_size: int,
    features: list[np.ndarray],
    labels: list[int],
    group_ids: list[int],
    group_to_id: dict[str, int],
    verbose: bool = False,
) -> None:
    buffers: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"HHAR CSV has no header: {path}")
        columns = {column.lower(): column for column in reader.fieldnames}
        required = {"x", "y", "z", "gt"}
        if not required.issubset(columns):
            raise ValueError(f"HHAR CSV missing columns {sorted(required)}: {path}")
        group_columns = [columns[name] for name in ["user", "model", "device"] if name in columns]
        for row_idx, row in enumerate(reader, 1):
            label_name = row[columns["gt"]].strip().lower().replace("_", "")
            if label_name not in HHAR_LABELS:
                if verbose and row_idx % 1_000_000 == 0:
                    print(f"[hhar] {path.name}: rows={row_idx} windows={len(features)}", flush=True)
                continue
            try:
                point = (
                    float(row[columns["x"]]),
                    float(row[columns["y"]]),
                    float(row[columns["z"]]),
                )
            except (TypeError, ValueError):
                if verbose and row_idx % 1_000_000 == 0:
                    print(f"[hhar] {path.name}: rows={row_idx} windows={len(features)}", flush=True)
                continue
            if group_columns:
                group_key = "|".join(row[column] for column in group_columns)
            else:
                group_key = path.stem
            if group_key not in group_to_id:
                group_to_id[group_key] = len(group_to_id)
            key = (group_key, label_name)
            buffer = buffers.setdefault(key, [])
            buffer.append(point)
            if len(buffer) >= window_size:
                window = np.asarray(buffer[:window_size], dtype=np.float32)
                features.append(_hhar_window_features(window))
                labels.append(HHAR_LABELS[label_name])
                group_ids.append(group_to_id[group_key])
                del buffer[:step_size]
            if verbose and row_idx % 1_000_000 == 0:
                print(f"[hhar] {path.name}: rows={row_idx} windows={len(features)}", flush=True)


def _hhar_dataset(data_dir: Path, seed: int):
    features, labels, _, num_classes = load_hhar_arrays(data_dir)
    rng = np.random.default_rng(seed)
    test_indices = []
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        test_size = max(1, int(round(0.2 * len(label_indices))))
        test_indices.extend(label_indices[:test_size].tolist())
    test_indices = np.asarray(sorted(test_indices), dtype=np.int64)
    test_mask = np.zeros(len(labels), dtype=bool)
    test_mask[test_indices] = True
    train_indices = np.where(~test_mask)[0]
    return (
        TensorDataset(torch.from_numpy(features[train_indices]), torch.from_numpy(labels[train_indices])),
        TensorDataset(torch.from_numpy(features[test_indices]), torch.from_numpy(labels[test_indices])),
        num_classes,
    )


def _hhar_window_features(window: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
        ]
    ).astype(np.float32)


def _hhar_root(data_dir: Path) -> Path:
    for name in ["HHAR", "hhar", "Activity recognition exp"]:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    return data_dir
