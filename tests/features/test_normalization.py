from unittest import TestCase
import pandas as pd

from damage_identification.features.normalization import Normalization

class TestNormalization(TestCase):
    def test_normalize_features(self):
        train_data = pd.DataFrame(
            {
                "first_n_samples": [1, 10],
                "peak_amplitude": [1,10],
                "counts": [1,10],
                "duration": [1,10],
                "rise_time": [1,10],
                "energy": [1,10],
                "peak_frequency": [1, 10],
                "central_frequency": [1, 10],
            }
        )

        test_data = pd.DataFrame(
            {
                "first_n_samples": [8], #
                "peak_amplitude": [8],
                "counts": [8],
                "duration": [8],
                "rise_time": [8],
                "energy": [8],
                "peak_frequency": [8],
                "central_frequency": [8],
            }
        )

        confirmation_data = pd.DataFrame(
            {
                "first_n_samples": [0.55555],
                "peak_amplitude": [0.55555],
                "counts": [0.55555],
                "duration": [0.55555],
                "rise_time": [0.55555],
                "energy": [0.55555],
                "peak_frequency": [0.55555],
                "central_frequency": [0.55555],
            }
        )

        normalized_features = Normalization()
        normalized_features.train(train_data)
        transformed_data = normalized_features.transform(test_data)
        self.assertTrue(((transformed_data - confirmation_data).abs() < 0.00001).all(axis=None))
