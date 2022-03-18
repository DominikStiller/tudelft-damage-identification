from unittest import TestCase

from damage_identification.clustering.optimal_number_clusters import find_optimal_number_of_clusters
from sklearn.datasets import make_blobs


class TestOptimalNumberOfClusters(TestCase):
    def test_optimal_number_of_clusters_kmeans(self):
        dummy_data, _ = make_blobs(n_samples=100, centers=3, n_features=30, random_state=0)
        clusternumber = find_optimal_number_of_clusters(dummy_data, 3, 6)
        self.assertEqual(clusternumber["Clusters kmeans"], 3)

    def test_optimal_number_of_clusters_hierarchical(self):
        dummy_data, _ = make_blobs(n_samples=100, centers=4, n_features=30, random_state=0)
        clusternumber = find_optimal_number_of_clusters(dummy_data, 3, 6)
        self.assertEqual(clusternumber["Clusters hierarchical"], 4)
