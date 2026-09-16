import unittest

import numpy as np

from iotlab_dfl.model import cosine_learning_rate, initialize_model, loss_and_gradient, parameter_count


class ModelTest(unittest.TestCase):
    def test_shape_and_finite_gradient(self):
        vector = initialize_model(12, 8, 3, 4)
        self.assertEqual(vector.size, parameter_count(12, 8, 3))
        rng = np.random.RandomState(2)
        x = rng.randn(5, 12).astype(np.float32)
        y = np.asarray([0, 1, 2, 1, 0], dtype=np.int64)
        loss, gradient = loss_and_gradient(vector, x, y, 12, 8, 3)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(gradient)))

    def test_gradient_matches_finite_difference(self):
        vector = initialize_model(3, 4, 2, 7)
        x = np.asarray([[0.2, -0.3, 0.7], [0.4, 0.1, -0.2]], dtype=np.float32)
        y = np.asarray([0, 1], dtype=np.int64)
        _, gradient = loss_and_gradient(vector, x, y, 3, 4, 2)
        epsilon = 1e-3
        for index in [0, 5, vector.size - 1]:
            left = vector.copy()
            right = vector.copy()
            left[index] -= epsilon
            right[index] += epsilon
            loss_left, _ = loss_and_gradient(left, x, y, 3, 4, 2)
            loss_right, _ = loss_and_gradient(right, x, y, 3, 4, 2)
            numerical = (loss_right - loss_left) / (2.0 * epsilon)
            self.assertAlmostEqual(float(gradient[index]), numerical, places=3)

    def test_two_hidden_layer_gradient(self):
        vector = initialize_model(5, [6, 4], 3, 9)
        self.assertEqual(vector.size, parameter_count(5, [6, 4], 3))
        rng = np.random.RandomState(8)
        x = rng.randn(4, 5).astype(np.float32)
        y = np.asarray([0, 1, 2, 1], dtype=np.int64)
        loss, gradient = loss_and_gradient(vector, x, y, 5, [6, 4], 3)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(gradient)))

        epsilon = 1e-3
        for index in [0, 17, vector.size - 1]:
            left, right = vector.copy(), vector.copy()
            left[index] -= epsilon
            right[index] += epsilon
            loss_left, _ = loss_and_gradient(left, x, y, 5, [6, 4], 3)
            loss_right, _ = loss_and_gradient(right, x, y, 5, [6, 4], 3)
            numerical = (loss_right - loss_left) / (2.0 * epsilon)
            self.assertAlmostEqual(float(gradient[index]), numerical, places=3)

    def test_cosine_learning_rate_endpoints(self):
        self.assertAlmostEqual(cosine_learning_rate(0.02, 0.002, 1, 150), 0.02)
        self.assertAlmostEqual(cosine_learning_rate(0.02, 0.002, 150, 150), 0.002)


if __name__ == "__main__":
    unittest.main()
