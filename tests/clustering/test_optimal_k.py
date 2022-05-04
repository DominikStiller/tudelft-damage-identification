from unittest import TestCase

from sklearn.datasets import make_blobs

from damage_identification.clustering.optimal_k import find_optimal_number_of_clusters


class TestOptimalNumberOfClusters(TestCase):
    def test_optimal_number_of_clusters(self):
        dummy_data, _ = make_blobs(n_samples=100, centers=3, n_features=30, random_state=0)
        clusternumber = find_optimal_number_of_clusters(dummy_data, 3, 6)
        self.assertEqual(clusternumber, 3)
