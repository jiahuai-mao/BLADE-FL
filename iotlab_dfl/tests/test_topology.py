import unittest

from iotlab_dfl.topology import sparse_regular_graph


class TopologyTest(unittest.TestCase):
    def test_degree_six_graph(self):
        graph = sparse_regular_graph(range(28), 6)
        self.assertEqual(len(graph), 28)
        self.assertTrue(all(len(neighbors) == 6 for neighbors in graph.values()))
        self.assertTrue(all(node in graph[neighbor] for node, neighbors in graph.items() for neighbor in neighbors))


if __name__ == "__main__":
    unittest.main()
