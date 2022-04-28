from unittest import TestCase

import numpy as np

from damage_identification.features.direct import DirectFeatureExtractor


class TestDirectFeatureExtractor(TestCase):
    def setUp(self):
        self.extractor = DirectFeatureExtractor(
            params={
                "direct_features_threshold": 0.5,
                "direct_features_n_samples": 2,
                "max_relative_peak_amplitude": 0.5,
                "first_peak_domain": 0.2,
                "sampling_rate": 1000 * 10,
            }
        )

    def test_extract_features_first_n_samples(self):
        example_1 = np.array([0, 1, 0, -2, 0, 1, 0, 0.4, -0.4, 0])
        features = self.extractor.extract_features(example_1)
        self.assertEqual(features["sample_1"], 0)
        self.assertEqual(features["sample_2"], 1)
        self.assertNotIn("sample_3", features)

    def test_extract_features_peak_amplitude(self):
        example_1 = np.array([0, 1, 0, -2, 0, 1, 0, 0.4, -0.4, 0])
        features = self.extractor.extract_features(example_1)
        # Expect 2 since peak amplitude is maximum *absolute* value
        self.assertEqual(features["peak_amplitude"], 2)

    def test_extract_features_counts(self):
        example_1 = np.array([0, 1, 0, -2, 0, 1, 0, 0.4, -0.4, 0])
        features = self.extractor.extract_features(example_1)
        self.assertEqual(features["counts"], 3)

    def test_extract_features_duration(self):
        example_1 = np.array([0, 1, 0, -2, 0, 1, 0, 0.4, -0.4, 0])
        features = self.extractor.extract_features(example_1)
        self.assertEqual(features["duration"], 4 / 10 / 1000)

    def test_extract_features_zero_duration_single_count(self):
        # If there is only a single count above the threshold, duration should be zero
        example_1 = np.array([0, 0.1, -0.1, 2, 0.3, 0, 0, 0, 0, 0])
        features = self.extractor.extract_features(example_1)
        self.assertEqual(features["counts"], 1)
        self.assertEqual(features["duration"], 0)

    def test_extract_features_rise_time(self):
        example_1 = np.array([0, 1, 0, -2, 0, 1, 0, 0.4, -0.4, 0])
        features = self.extractor.extract_features(example_1)
        self.assertEqual(features["rise_time"], 2 / 10 / 1000)

    def test_extract_features_2peaks(self):
        example_1 = np.array([0, 2, 0, -1, 0, 1, 0, 1.8, -0.4, 0])
        features = self.extractor.extract_features(example_1)
        self.assertEqual(features["duration"], None)
