from __future__ import annotations

import gzip
import pickle
import struct
from pathlib import Path
from typing import Sequence

import numpy as np

from .types import ClientMetadata


def load_labels(dataset: str, data_dir: str | Path) -> tuple[np.ndarray, int]:
    dataset = dataset.lower()
    data_dir = Path(data_dir)
    if dataset == "synthetic":
        labels = np.tile(np.arange(10, dtype=np.int64), 100)
        return labels, 10
    if dataset == "mnist":
        return _load_mnist_labels(data_dir), 10
    if dataset == "cifar10":
        return _load_cifar10_labels(data_dir), 10
    if dataset == "cifar100":
        return _load_cifar100_labels(data_dir), 100
    if dataset == "hhar":
        return _load_hhar_labels(data_dir), 6
    if dataset in {"fashionmnist", "svhn"}:
        return _load_torchvision_labels(dataset, data_dir)
    raise ValueError(f"Unsupported dataset: {dataset}")


def partition_iid(
    labels: Sequence[int],
    num_clients: int,
    samples_fraction: float = 1.0,
    seed: int = 0,
) -> list[list[int]]:
    _validate_partition_args(labels, num_clients, samples_fraction)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    if samples_fraction < 1.0:
        size = max(num_clients, int(round(len(indices) * samples_fraction)))
        indices = rng.choice(indices, size=size, replace=False)
    rng.shuffle(indices)
    return [split.astype(np.int64).tolist() for split in np.array_split(indices, num_clients)]


def partition_dirichlet(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    samples_fraction: float = 1.0,
    seed: int = 0,
    min_size: int = 1,
) -> list[list[int]]:
    _validate_partition_args(labels, num_clients, samples_fraction)
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    labels_arr = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = np.arange(len(labels_arr))
    if samples_fraction < 1.0:
        size = max(num_clients, int(round(len(selected) * samples_fraction)))
        selected = rng.choice(selected, size=size, replace=False)
    selected_set = set(selected.tolist())

    classes = np.unique(labels_arr[selected])
    for _ in range(1000):
        client_indices = [[] for _ in range(num_clients)]
        for cls in classes:
            class_indices = np.where(labels_arr == cls)[0]
            class_indices = np.asarray([idx for idx in class_indices if int(idx) in selected_set], dtype=np.int64)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for cid, part in enumerate(np.split(class_indices, split_points)):
                client_indices[cid].extend(part.astype(np.int64).tolist())
        for indices in client_indices:
            rng.shuffle(indices)
        if min(len(indices) for indices in client_indices) >= min_size:
            return client_indices
    raise RuntimeError("failed to build a non-empty Dirichlet partition after 1000 attempts.")


def build_client_metadata(
    client_indices: Sequence[Sequence[int]],
    labels: Sequence[int],
    num_classes: int,
    time_mode: str = "num_samples",
    local_epochs: float = 1.0,
) -> list[ClientMetadata]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    metadata = []
    for cid, indices in enumerate(client_indices):
        idx = np.asarray(indices, dtype=np.int64)
        counts = np.bincount(labels_arr[idx], minlength=num_classes).astype(int).tolist()
        num_samples = int(len(idx))
        if time_mode == "num_samples":
            train_time = float(num_samples)
        elif time_mode == "local_epochs_num_samples":
            train_time = float(local_epochs) * float(num_samples)
        else:
            raise ValueError(f"Unsupported time_mode: {time_mode}")
        metadata.append(
            ClientMetadata(
                client_id=cid,
                num_samples=num_samples,
                label_counts=counts,
                train_time=train_time,
            )
        )
    return metadata


def _validate_partition_args(labels: Sequence[int], num_clients: int, samples_fraction: float) -> None:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if len(labels) < num_clients:
        raise ValueError("number of labels must be at least num_clients.")
    if not 0 < samples_fraction <= 1:
        raise ValueError("samples_fraction must be in (0, 1].")


def _load_mnist_labels(data_dir: Path) -> np.ndarray:
    candidates = [
        data_dir / "MNIST" / "raw" / "train-labels-idx1-ubyte",
        data_dir / "MNIST" / "raw" / "train-labels-idx1-ubyte.gz",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("MNIST train labels not found under data/MNIST/raw.")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file magic: {magic}")
        labels = np.frombuffer(f.read(size), dtype=np.uint8).astype(np.int64)
    return labels


def _load_cifar10_labels(data_dir: Path) -> np.ndarray:
    root = data_dir / "cifar-10-batches-py"
    labels = []
    for idx in range(1, 6):
        path = root / f"data_batch_{idx}"
        if not path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10 batch: {path}")
        with path.open("rb") as f:
            payload = pickle.load(f, encoding="latin1")
        labels.extend(payload.get("labels", payload.get(b"labels")))
    return np.asarray(labels, dtype=np.int64)


def _load_cifar100_labels(data_dir: Path) -> np.ndarray:
    path = data_dir / "cifar-100-python" / "train"
    if not path.exists():
        raise FileNotFoundError(f"Missing CIFAR-100 train file: {path}")
    with path.open("rb") as f:
        payload = pickle.load(f, encoding="latin1")
    labels = payload.get("fine_labels", payload.get(b"fine_labels"))
    return np.asarray(labels, dtype=np.int64)


def _load_torchvision_labels(dataset: str, data_dir: Path) -> tuple[np.ndarray, int]:
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise RuntimeError(f"torchvision is required to load {dataset}.") from exc

    if dataset == "fashionmnist":
        ds = datasets.FashionMNIST(root=str(data_dir), train=True, download=False)
        return np.asarray(ds.targets, dtype=np.int64), 10
    if dataset == "svhn":
        ds = datasets.SVHN(root=str(data_dir), split="train", download=False)
        labels = np.asarray(ds.labels, dtype=np.int64)
        return labels, 10
    raise ValueError(f"Unsupported torchvision dataset: {dataset}")


def _load_hhar_labels(data_dir: Path) -> np.ndarray:
    try:
        from clustered_dfl.training.data_loaders import load_hhar_arrays
    except ImportError as exc:
        raise RuntimeError("HHAR label loading requires the training data loader module.") from exc
    _, labels, _, _ = load_hhar_arrays(data_dir)
    return labels
