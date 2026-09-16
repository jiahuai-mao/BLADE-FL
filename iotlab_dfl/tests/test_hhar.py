import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from iotlab_dfl.hhar import load_hhar_stats
from iotlab_dfl.prepare_hhar import (
    block_dirichlet_partition,
    rich48_feature_names,
    rich48_features,
    temporal_blocks,
)


class HHARTest(unittest.TestCase):
    def test_streaming_features_groups_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "HHAR"
            root.mkdir()
            path = root / "Phones_accelerometer.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["user", "model", "device", "x", "y", "z", "gt"])
                writer.writeheader()
                for user, label, offset in [("a", "walk", 0.0), ("b", "sit", 10.0)]:
                    for index in range(6):
                        writer.writerow(
                            {
                                "user": user,
                                "model": "m",
                                "device": "d",
                                "x": offset + index,
                                "y": offset + index + 1,
                                "z": offset + index + 2,
                                "gt": label,
                            }
                        )
            first = load_hhar_stats(Path(directory), window_size=4, step_size=2, verbose=False)
            second = load_hhar_stats(Path(directory), window_size=4, step_size=2, verbose=False)
            features, labels, groups, classes = first
            self.assertEqual(features.shape, (4, 12))
            self.assertEqual(classes, 6)
            self.assertEqual(set(labels.tolist()), {1, 3})
            self.assertEqual(set(groups.tolist()), {0, 1})
            np.testing.assert_allclose(features, second[0])

    def test_rich48_shape_and_finite_values(self):
        rng = np.random.RandomState(5)
        windows = rng.randn(3, 6, 16).astype(np.float32)
        features = rich48_features(windows)
        self.assertEqual(features.shape, (3, 48))
        self.assertEqual(len(rich48_feature_names()), 48)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_temporal_block_partition_is_complete_and_reproducible(self):
        labels = np.tile(np.repeat(np.arange(3), 80), 3).astype(np.int64)
        groups = np.repeat(np.arange(3), 240).astype(np.int64)
        source = np.arange(len(labels), dtype=np.int64)
        kwargs = dict(
            labels=labels,
            group_ids=groups,
            source_indices=source,
            num_clients=6,
            num_classes=3,
            alpha=0.3,
            block_size=32,
            min_samples=60,
            min_labels=2,
            min_label_samples=8,
            max_sample_factor=2.5,
            seed=7,
        )
        first, blocks, _ = block_dirichlet_partition(**kwargs)
        second, _, _ = block_dirichlet_partition(**kwargs)
        self.assertEqual([item.tolist() for item in first], [item.tolist() for item in second])
        self.assertEqual(sorted(np.concatenate(first).tolist()), list(range(len(labels))))
        self.assertGreater(len(blocks), 3)
        for selected in first:
            self.assertGreaterEqual(len(selected), 60)
            label_counts = np.bincount(labels[selected], minlength=3)
            self.assertGreaterEqual(int(np.sum(label_counts >= 8)), 2)

    def test_temporal_blocks_do_not_cross_group_or_label_runs(self):
        labels = np.asarray([0, 0, 0, 1, 1, 0, 0], dtype=np.int64)
        groups = np.asarray([4, 4, 4, 4, 4, 5, 5], dtype=np.int64)
        blocks = temporal_blocks(labels, groups, np.arange(7), maximum_size=2)
        for block in blocks:
            selected = block["indices"]
            self.assertEqual(len(np.unique(labels[selected])), 1)
            self.assertEqual(len(np.unique(groups[selected])), 1)
            self.assertLessEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
