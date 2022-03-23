from unittest import TestCase
import numpy as np
from damage_identification.clustering.fcmeans import FCMeansClusterer


class TestFCMeans(TestCase):
    def test_pca(self):
        test_point = np.array([[0, 0], [12, 3]])
        test_set = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [12, 3]])

        model = FCMeansClusterer({"n_clusters": 3})
        model.train(test_set)
        result = model.predict(test_point)
        print(result)
        self.assertEqual(result, np.array([0, 1]))