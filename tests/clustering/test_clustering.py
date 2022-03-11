from unittest import TestCase
from numpy import testing

import numpy as np
import os

from damage_identification.clustering.kmeans import KmeansClustering


class TestKmeansClustering(TestCase):
    def test_kmeans_clustering(self):
        test_set = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        kmeans_model = KmeansClustering(2).train(test_set)
        testing.assert_array_equal(kmeans_model.labels_, [1, 1, 1, 0, 0, 0])
        testing.assert_array_equal(kmeans_model.cluster_centers_, [[10.,  2.], [1.,  2.]])


class TestKmeansPredict(TestCase):
    def test_kmeans_prediction(self):
        test_point = np.array([[0, 0], [12, 3]])
        test_set = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        kmeans_model = KmeansClustering(2).train(test_set)
        parent_cluster = kmeans_model.predict(test_point)
        testing.assert_array_equal(parent_cluster, np.array([1, 0]))


class TestDumpFileLoading(TestCase):
    def test_dump_file_load(self):
        test_point = np.array([[0, 0], [12, 3]])
        test_set = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        kmeans = KmeansClustering(2)
        kmeans.train(test_set)
        directory = os.path.dirname(os.path.abspath(__file__))
        kmeans.save(directory)
        kmeans_model = KmeansClustering(2)
        kmeans_model.load(directory)
        point_prediction = kmeans_model.predict(test_point)
        testing.assert_array_equal(point_prediction, np.array([1, 0]))
