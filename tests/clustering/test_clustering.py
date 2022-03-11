from unittest import TestCase
from numpy import testing

import numpy as np
import os

from damage_identification.clustering.kmeans import KmeansClustering


class TestKmeansClustering(TestCase):
    def test_kmeans_clustering(self):
        test_set = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        kmeans_model, point_labels, clusters_loc = KmeansClustering("cluster1", 2).train(test_set)
        testing.assert_array_equal(point_labels, [1, 1, 1, 0, 0, 0])
        testing.assert_array_equal(clusters_loc, [[10.,  2.], [1.,  2.]])


class TestKmeansPredict(TestCase):
    def test_kmeans_prediction(self):
        test_point = np.array([[0, 0], [12, 3]])
        test_set = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        kmeans_model, point_labels, clusters_loc = KmeansClustering("cluster1", 2).train(test_set)
        print(kmeans_model)
        parent_cluster = kmeans_model.predict(test_point)
        print(parent_cluster)
        testing.assert_array_equal(parent_cluster, np.array([1, 0]))


class TestDumpFileLoading(TestCase):
    def test_dump_file_load(self):
        test_point = np.array([[0, 0], [12, 3]])
        test_set = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        KmeansClustering("cluster2", 2).train(test_set)
        directory = os.path.dirname(os.path.abspath(__file__))
        KmeansClustering("cluster2", 2).save(directory)
        kmeans_model = KmeansClustering("cluster2", 2).load(directory)
        #print(kmeans_model)
        point_prediction = kmeans_model.predict(test_point)
        testing.assert_array_equal(point_prediction, np.array([1, 0]))
