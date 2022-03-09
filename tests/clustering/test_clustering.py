from unittest import TestCase
from numpy import testing

import numpy as np

from damage_identification.clustering.kmeans import KmeansClustering


class TestKmeansClustering(TestCase):
    def test_kmeans_clustering(self):
        test_set = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        point_labels, clusters_loc = KmeansClustering("cluster1", 2).train(test_set)
        testing.assert_array_equal(point_labels, [1, 1, 1, 0, 0, 0])
        testing.assert_array_equal(clusters_loc, [[10.,  2.], [1.,  2.]])


class TestKmeansPredict(TestCase):
    def test_kmeans_prediction(self):
        test_point = np.array([[0, 0], [12, 3]])
        parent_cluster = KmeansClustering("cluster1", 2).predict(test_point)
        self.assertEqual(parent_cluster, np.array([1, 0]))

