import tempfile
from unittest import TestCase

import numpy as np
from numpy import testing

from damage_identification.clustering.hierarchical import HierarchicalClusterer


class TestHierarchicalClusterer(TestCase):
    def test_hierarchical_clustering(self):
        test_set = np.array(
            [
                [1],
                [2],
                [3],
                [4],
                [5],
                [11],
                [12],
                [13],
                [14],
                [15],
                [21],
                [22],
                [23],
                [24],
                [25],
                [31],
                [32],
                [33],
                [34],
                [35],
            ]
        )

        hierarchical_model = HierarchicalClusterer({"n_clusters": 4, "n_neighbors": 5})
        testing.assert_array_equal(
            hierarchical_model._do_hierarchical_clustering(test_set),
            (3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0),
        )

    def test_hierarchical_prediction(self):
        test_set = np.array(
            [
                [1],
                [2],
                [3],
                [4],
                [5],
                [11],
                [12],
                [13],
                [14],
                [15],
                [21],
                [22],
                [23],
                [24],
                [25],
                [31],
                [32],
                [33],
                [34],
                [35],
            ]
        )
        hierarchical_model = HierarchicalClusterer({"n_clusters": 4, "n_neighbors": 5})
        hierarchical_model.train(test_set)
        self.assertEqual(hierarchical_model.predict(np.array([[6]])), 3)
        self.assertEqual(hierarchical_model.predict(np.array([[29]])), 0)

    def test_dump_file_load(self):
        test_point = [[1]]
        test_set = np.array(
            [
                [1],
                [2],
                [3],
                [4],
                [5],
                [11],
                [12],
                [13],
                [14],
                [15],
                [21],
                [22],
                [23],
                [24],
                [25],
                [31],
                [32],
                [33],
                [34],
                [35],
            ]
        )

        hierarchical_model = HierarchicalClusterer({"n_clusters": 4, "n_neighbors": 5})
        hierarchical_model.train(test_set)

        with tempfile.TemporaryDirectory() as tmpdir:
            hierarchical_model.save(tmpdir)
            hierarchical_model = HierarchicalClusterer({})
            hierarchical_model.load(tmpdir)

        point_prediction = hierarchical_model.predict(test_point)
        testing.assert_array_equal(point_prediction, 3)
