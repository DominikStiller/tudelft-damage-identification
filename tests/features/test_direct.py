from unittest import TestCase

import numpy as np

from damage_identification.features.direct import DirectFeatureExtractor


class TestDirectFeatureExtractor(TestCase):
    def test_extract_features_peak_amplitude(self):
        example_1 = np.array([0, 1, 0, -2, 0, 1, 0, 0.4, -0.4, 0])
        features = DirectFeatureExtractor().extract_features(example_1)
        # Expect 10 since peak amplitude is maximum *absolute* value
        self.assertEqual(features["peak_amplitude"], 2)

    def test_extract_features_counts(self):
        example_1 = np.array([0, 1, 0, -2, 0, 1, 0, 0.4, -0.4, 0])
        features = DirectFeatureExtractor().extract_features(example_1)
        self.assertEqual(features["count"], 3)

    def test_extract_features_duration(self):
        example_1 = np.array([0, 1, 0, -2, 0, 1, 0, 0.4, -0.4, 0])
        features = DirectFeatureExtractor().extract_features(example_1)
        self.assertEqual(features["duration"], 4/10/1000)

    def test_extract_features_rise_time(self):
        example_1 = np.array([0, 1, 0, -2, 0, 1, 0, 0.4, -0.4, 0])
        features = DirectFeatureExtractor().extract_features(example_1)
        self.assertEqual(features["rise_time"], 2/10/1000)
    
