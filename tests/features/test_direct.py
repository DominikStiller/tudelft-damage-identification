from unittest import TestCase

import numpy as np

from damage_identification.features.direct import DirectFeatureExtractor


class TestDirectFeatureExtractor(TestCase):
    def test_extract_features_peak_amplitude(self):
        example = np.array([5, 3.4, -10, 6.4, 0])
        features = DirectFeatureExtractor().extract_features(example)
        # Expect 10 since peak amplitude is maximum *absolute* value
        self.assertEqual(features["peak_amplitude"], 10)
        self.assertEqual(5, 10)
