from unittest import TestCase
import pandas as pd

from damage_identification.features.normalization import Normalization


class TestNormalization(TestCase):
    def test_normalize_features(self):
        train_data = pd.DataFrame({"test": [1, 10]})

        test_data = pd.DataFrame({"test": [8]})

        confirmation_data = pd.DataFrame({"test": [0.55555]})

        normalized_features = Normalization()
        normalized_features.train(train_data)
        transformed_data = normalized_features.transform(test_data)
        self.assertTrue(((transformed_data - confirmation_data).abs() < 0.00001).all(axis=None))
