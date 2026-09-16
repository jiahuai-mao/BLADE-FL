from __future__ import annotations

import csv
import gzip
import hashlib
import os
import pickle
import struct
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from .base import ClientState
from clustered_dfl.types import ClientMetadata


def load_torch_datasets(
    dataset: str,
    data_dir: str | Path,
    seed: int = 0,
    hhar_feature_mode: str = "stats",
    hhar_window_size: int = 128,
    hhar_step_size: int = 64,
    hhar_sensor_mode: str = "auto",
    hhar_test_split: str = "window",
):
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
        return _hhar_dataset(
            data_dir,
            seed,
            feature_mode=hhar_feature_mode,
            window_size=hhar_window_size,
            step_size=hhar_step_size,
            sensor_mode=hhar_sensor_mode,
            test_split=hhar_test_split,
        )
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
    device: str = "cpu",
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
            pin_memory=str(device).startswith("cuda"),
            persistent_workers=False,
        )
        states.append(
            ClientState(
                client_id=cid,
                train_loader=loader,
                metadata=metadata_by_id[cid],
            )
        )
    return states


def make_test_loader(test_dataset, batch_size: int, num_workers: int = 0, device: str = "cpu") -> DataLoader:
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=num_workers > 0,
    )


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
    return (
        CIFARArrayDataset(np.concatenate(train_images, axis=0), train_labels, train=True),
        CIFARArrayDataset(test_payload["data"], test_payload["labels"], train=False),
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
    arr = np.asarray(flat_data, dtype=np.uint8).reshape(-1, 3, 32, 32)
    buffer = bytearray(arr.tobytes())
    tensor = torch.frombuffer(buffer, dtype=torch.uint8).reshape(-1, 3, 32, 32)
    return tensor.float() / 255.0


CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).reshape(3, 1, 1)
CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2616), dtype=torch.float32).reshape(3, 1, 1)


class CIFARArrayDataset(torch.utils.data.Dataset):
    def __init__(self, flat_data, labels, train: bool) -> None:
        images = np.asarray(flat_data, dtype=np.uint8).reshape(-1, 3, 32, 32)
        self._image_buffer = bytearray(images.tobytes())
        self.images = torch.frombuffer(self._image_buffer, dtype=torch.uint8).reshape(-1, 3, 32, 32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.train = train

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int):
        image = self.images[index].float() / 255.0
        if self.train:
            image = _random_crop_32(image, padding=4)
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(2,))
        image = (image - CIFAR10_MEAN) / CIFAR10_STD
        return image, self.labels[index]


def _random_crop_32(image: torch.Tensor, padding: int = 4) -> torch.Tensor:
    padded = F.pad(image, (padding, padding, padding, padding))
    top = int(torch.randint(0, 2 * padding + 1, (1,)).item())
    left = int(torch.randint(0, 2 * padding + 1, (1,)).item())
    return padded[:, top : top + 32, left : left + 32]


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
    step_size: int = 64,
    feature_mode: str = "stats",
    sensor_mode: str = "auto",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if window_size <= 0:
        raise ValueError("HHAR window_size must be positive.")
    if step_size <= 0:
        raise ValueError("HHAR step_size must be positive.")
    feature_mode = _validate_hhar_feature_mode(feature_mode)
    sensor_mode = _validate_hhar_sensor_mode(sensor_mode)
    root = _hhar_root(data_dir)
    accel_paths = _hhar_sensor_paths(root, "accelerometer")
    gyro_paths = _hhar_sensor_paths(root, "gyroscope")
    if not accel_paths:
        raise FileNotFoundError(
            "HHAR accelerometer CSV files not found. Expected data/HHAR/Phones_accelerometer.csv "
            "or data/hhar/Watch_accelerometer.csv."
        )
    if sensor_mode == "acc_gyro" and not gyro_paths:
        raise FileNotFoundError(
            "HHAR gyroscope CSV files not found, but --hhar-sensor-mode acc_gyro was requested. "
            "Expected files such as data/HHAR/Phones_gyroscope.csv and data/HHAR/Watch_gyroscope.csv."
        )
    use_gyro = sensor_mode == "acc_gyro" or (sensor_mode == "auto" and bool(gyro_paths))
    if verbose:
        accel_names = ", ".join(path.name for path in accel_paths)
        print(f"[hhar] loading accelerometer CSV files from {root}: {accel_names}", flush=True)
        if use_gyro:
            gyro_names = ", ".join(path.name for path in gyro_paths)
            print(f"[hhar] loading gyroscope CSV files from {root}: {gyro_names}", flush=True)
        else:
            print("[hhar] gyroscope CSV files not found; using accelerometer-only 3-channel windows.", flush=True)

    active_gyro_paths = gyro_paths if use_gyro else []
    cache_path = _hhar_cache_path(root, accel_paths, active_gyro_paths, window_size, step_size, feature_mode, sensor_mode)
    cached = _load_hhar_cache(cache_path, verbose)
    if cached is not None:
        return cached

    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        pd = None

    if pd is None or not use_gyro:
        result = _load_hhar_accel_only_arrays(
            accel_paths,
            window_size,
            step_size,
            feature_mode,
            verbose,
        )
        _save_hhar_cache(cache_path, result, verbose)
        return result

    accel_windows, group_to_id = _load_hhar_sensor_windows(
        accel_paths,
        window_size,
        step_size,
        feature_mode,
        verbose,
        "accelerometer",
    )
    gyro_windows, group_to_id = _load_hhar_sensor_windows(
        gyro_paths,
        window_size,
        step_size,
        feature_mode,
        verbose,
        "gyroscope",
        group_to_id,
    )

    features = []
    labels = []
    group_ids = []
    paired = 0
    for key in sorted(accel_windows):
        if key not in gyro_windows:
            continue
        group_key, label_name = key
        acc_items = accel_windows[key]
        gyro_items = gyro_windows[key]
        count = min(len(acc_items), len(gyro_items))
        group_id = group_to_id[group_key]
        for idx in range(count):
            if feature_mode == "raw":
                feature = np.concatenate([acc_items[idx], gyro_items[idx]], axis=0)
            else:
                feature = np.concatenate([acc_items[idx], gyro_items[idx]], axis=0)
            features.append(feature)
            labels.append(HHAR_LABELS[label_name])
            group_ids.append(group_id)
        paired += count
    if not features:
        raise RuntimeError("HHAR accelerometer/gyroscope CSV files were found, but no paired labelled windows could be constructed.")
    if verbose:
        print(f"[hhar] total paired windows={paired} groups={len(group_to_id)} channels=6", flush=True)
    result = (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
        len(HHAR_LABELS),
    )
    _save_hhar_cache(cache_path, result, verbose)
    return result


def _hhar_cache_path(
    root: Path,
    accel_paths: list[Path],
    gyro_paths: list[Path],
    window_size: int,
    step_size: int,
    feature_mode: str,
    sensor_mode: str,
) -> Path:
    digest = hashlib.sha1()
    digest.update(f"window={window_size}|step={step_size}|mode={feature_mode}|sensor={sensor_mode}".encode("utf-8"))
    for path in accel_paths + gyro_paths:
        stat = path.stat()
        digest.update(f"|{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    return root / ".cache" / f"hhar_{digest.hexdigest()[:16]}.npz"


def _load_hhar_cache(cache_path: Path, verbose: bool):
    if not cache_path.exists():
        return None
    if verbose:
        print(f"[hhar] loading cached windows from {cache_path}", flush=True)
    with np.load(cache_path) as payload:
        return (
            payload["features"].astype(np.float32, copy=False),
            payload["labels"].astype(np.int64, copy=False),
            payload["group_ids"].astype(np.int64, copy=False),
            int(payload["num_classes"]),
        )


def _save_hhar_cache(cache_path: Path, result, verbose: bool) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
    features, labels, group_ids, num_classes = result
    np.savez(
        tmp_path,
        features=features,
        labels=labels,
        group_ids=group_ids,
        num_classes=np.asarray(num_classes, dtype=np.int64),
    )
    tmp_path.replace(cache_path)
    if verbose:
        print(f"[hhar] cached windows at {cache_path}", flush=True)


def _load_hhar_accel_only_arrays(
    paths: list[Path],
    window_size: int,
    step_size: int,
    feature_mode: str,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:

    features = []
    labels = []
    group_ids = []
    group_to_id: dict[str, int] = {}
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
                feature_mode,
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
                features.append(_hhar_window_features(window, feature_mode))
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


def _hhar_sensor_paths(root: Path, sensor: str) -> list[Path]:
    candidates = [
        root / f"Phones_{sensor}.csv",
        root / f"Watch_{sensor}.csv",
        root / f"phones_{sensor}.csv",
        root / f"watch_{sensor}.csv",
    ]
    paths = [path for path in candidates if path.exists()]
    if paths:
        return paths
    return [path for path in sorted(root.glob("*.csv")) if sensor in path.name.lower()]


def _load_hhar_sensor_windows(
    paths: list[Path],
    window_size: int,
    step_size: int,
    feature_mode: str,
    verbose: bool,
    sensor_name: str,
    group_to_id: dict[str, int] | None = None,
) -> tuple[dict[tuple[str, str], list[np.ndarray]], dict[str, int]]:
    import pandas as pd

    windows: dict[tuple[str, str], list[np.ndarray]] = {}
    group_to_id = {} if group_to_id is None else group_to_id
    for path in paths:
        before = sum(len(items) for items in windows.values())
        if verbose:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"[hhar] reading {path.name} ({size_mb:.1f} MB)...", flush=True)
        frame = pd.read_csv(path)
        columns = {column.lower(): column for column in frame.columns}
        required = {"x", "y", "z", "gt"}
        if not required.issubset(columns):
            raise ValueError(f"HHAR CSV missing columns {sorted(required)}: {path}")
        time_col = columns.get("creation_time") or columns.get("arrival_time")
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
        sort_cols = group_cols + ["_label"]
        if time_col is not None:
            sort_cols.append(time_col)
        frame = frame.sort_values(sort_cols)
        for group_values, group_frame in frame.groupby(group_cols + ["_label"], sort=True):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            label_name = str(group_values[-1])
            group_key = "|".join(str(value) for value in group_values[:-1])
            if group_key not in group_to_id:
                group_to_id[group_key] = len(group_to_id)
            values = group_frame[[columns["x"], columns["y"], columns["z"]]].to_numpy(dtype=np.float32)
            key = (group_key, label_name)
            bucket = windows.setdefault(key, [])
            for start in range(0, max(len(values) - window_size + 1, 0), step_size):
                window = values[start : start + window_size]
                bucket.append(_hhar_window_features(window, feature_mode))
        if verbose:
            after = sum(len(items) for items in windows.values())
            print(f"[hhar] finished {path.name}: {sensor_name}_windows={after - before}", flush=True)
    return windows, group_to_id


def _load_hhar_csv_streaming(
    path: Path,
    window_size: int,
    step_size: int,
    features: list[np.ndarray],
    labels: list[int],
    group_ids: list[int],
    group_to_id: dict[str, int],
    feature_mode: str,
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
                features.append(_hhar_window_features(window, feature_mode))
                labels.append(HHAR_LABELS[label_name])
                group_ids.append(group_to_id[group_key])
                del buffer[:step_size]
            if verbose and row_idx % 1_000_000 == 0:
                print(f"[hhar] {path.name}: rows={row_idx} windows={len(features)}", flush=True)


def _hhar_dataset(
    data_dir: Path,
    seed: int,
    feature_mode: str = "stats",
    window_size: int = 128,
    step_size: int = 64,
    sensor_mode: str = "auto",
    test_split: str = "window",
):
    feature_mode = _validate_hhar_feature_mode(feature_mode)
    test_split = _validate_hhar_test_split(test_split)
    features, labels, group_ids, num_classes = load_hhar_arrays(
        data_dir,
        window_size=window_size,
        step_size=step_size,
        feature_mode=feature_mode,
        sensor_mode=sensor_mode,
    )
    rng = np.random.default_rng(seed)
    if test_split == "group":
        test_mask = _hhar_group_test_mask(group_ids, rng, test_fraction=0.2)
    else:
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
    train_x = features[train_indices]
    test_x = features[test_indices]
    train_group_ids = group_ids[train_indices]
    test_group_ids = group_ids[test_indices]
    if feature_mode == "raw":
        mean = train_x.mean(axis=(0, 2), keepdims=True)
        std = train_x.std(axis=(0, 2), keepdims=True)
        train_x = (train_x - mean) / np.maximum(std, 1e-6)
        test_x = (test_x - mean) / np.maximum(std, 1e-6)
    train_dataset = TensorDataset(_float_tensor_from_numpy(train_x.astype(np.float32)), _long_tensor_from_numpy(labels[train_indices]))
    test_dataset = TensorDataset(_float_tensor_from_numpy(test_x.astype(np.float32)), _long_tensor_from_numpy(labels[test_indices]))
    train_dataset.hhar_group_ids = _long_tensor_from_numpy(train_group_ids)
    test_dataset.hhar_group_ids = _long_tensor_from_numpy(test_group_ids)
    return train_dataset, test_dataset, num_classes


def _hhar_window_features(window: np.ndarray, feature_mode: str = "stats") -> np.ndarray:
    if feature_mode == "raw":
        return window.T.astype(np.float32)
    return np.concatenate(
        [
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
        ]
    ).astype(np.float32)


def _validate_hhar_feature_mode(feature_mode: str) -> str:
    value = feature_mode.lower()
    if value not in {"stats", "raw"}:
        raise ValueError(f"Unsupported HHAR feature mode: {feature_mode}. Use 'stats' or 'raw'.")
    return value


def _validate_hhar_sensor_mode(sensor_mode: str) -> str:
    value = sensor_mode.lower()
    if value not in {"auto", "acc", "acc_gyro"}:
        raise ValueError(f"Unsupported HHAR sensor mode: {sensor_mode}. Use 'auto', 'acc', or 'acc_gyro'.")
    return value


def _validate_hhar_test_split(test_split: str) -> str:
    value = test_split.lower()
    if value not in {"window", "group"}:
        raise ValueError(f"Unsupported HHAR test split: {test_split}. Use 'window' or 'group'.")
    return value


def _hhar_group_test_mask(group_ids: np.ndarray, rng: np.random.Generator, test_fraction: float) -> np.ndarray:
    unique_groups = np.unique(group_ids)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    target = max(1, int(round(len(group_ids) * test_fraction)))
    selected = []
    selected_count = 0
    for group_id in shuffled.tolist():
        selected.append(group_id)
        selected_count += int(np.sum(group_ids == group_id))
        if selected_count >= target:
            break
    test_mask = np.isin(group_ids, np.asarray(selected, dtype=group_ids.dtype))
    if bool(test_mask.all()):
        raise RuntimeError("HHAR group split placed all windows in the test set.")
    return test_mask


def _float_tensor_from_numpy(array: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(array, dtype=np.float32)
    buffer = bytearray(array.tobytes())
    return torch.frombuffer(buffer, dtype=torch.float32).reshape(array.shape)


def _long_tensor_from_numpy(array: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(array, dtype=np.int64)
    buffer = bytearray(array.tobytes())
    return torch.frombuffer(buffer, dtype=torch.int64).reshape(array.shape)


def _hhar_root(data_dir: Path) -> Path:
    for name in ["HHAR", "hhar", "Activity recognition exp"]:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    return data_dir
