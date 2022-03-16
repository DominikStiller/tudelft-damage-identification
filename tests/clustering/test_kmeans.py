from unittest import TestCase
from numpy import testing

import numpy as np
import tempfile


from damage_identification.clustering.kmeans import KmeansClusterer


class TestKmeansClusterer(TestCase):
    def test_kmeans_clustering(self):
        test_set = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
                [3, 3, 3, 3, 3, 4],
                [4, 4, 4, 4, 4, 4],
                [7, 7, 7, 7, 7, 8],
                [8, 8, 8, 8, 8, 8],
                [11, 11, 11, 11, 11, 12],
                [12, 12, 12, 12, 12, 12],
                [13, 13, 13, 13, 13, 14],
                [14, 14, 14, 14, 14, 14],
                [17, 17, 17, 17, 17, 18],
                [18, 18, 18, 18, 18, 18],
            ]
        )

        kmeans_model = KmeansClusterer({"n_clusters": 6}).train(test_set)

        testing.assert_array_equal(kmeans_model.labels_, [2, 2, 4, 4, 0, 0, 3, 3, 5, 5, 1, 1])
        testing.assert_array_equal(
            kmeans_model.cluster_centers_,
            [
                [7.5, 7.5, 7.5, 7.5, 7.5, 8.0],
                [17.5, 17.5, 17.5, 17.5, 17.5, 18.0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
                [11.5, 11.5, 11.5, 11.5, 11.5, 12.0],
                [3.5, 3.5, 3.5, 3.5, 3.5, 4.0],
                [13.5, 13.5, 13.5, 13.5, 13.5, 14.0],
            ],
        )

    def test_kmeans_prediction(self):
        test_point = np.array([[0, 0], [12, 3]])
        test_set = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

        kmeans_model = KmeansClusterer({"n_clusters": 6}).train(test_set)
        parent_cluster = kmeans_model.predict(test_point)

        testing.assert_array_equal(parent_cluster, np.array([1, 0]))

    def test_dump_file_load(self):
        test_point = np.array([[12, 3]])
        test_set = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

        kmeans = KmeansClusterer({"n_clusters": 6})
        kmeans.train(test_set)

        with tempfile.TemporaryDirectory() as tmpdir:
            kmeans.save(tmpdir)
            kmeans_model = KmeansClusterer({})
            kmeans_model.load(tmpdir)

        point_prediction = kmeans_model.predict(test_point)
        testing.assert_array_equal(point_prediction, 0)
