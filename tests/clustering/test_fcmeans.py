from unittest import TestCase

import pandas as pd

from damage_identification.clustering.fcmeans import FCMeansClusterer


class TestFCMeans(TestCase):
    def test_fcmeans(self):
        train_set = pd.DataFrame([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [12, 3]])

        model = FCMeansClusterer({"n_clusters": 3})
        model.train(train_set)

        self.assertEqual(model.predict(pd.DataFrame([[0, 0]])), 0)
        self.assertEqual(model.predict(pd.DataFrame([[12, 3]])), 1)
