from unittest import TestCase
import numpy as np
from damage_identification.features.fourier import FourierExtractor


class TestFourierFeatureExtractor(TestCase):
    def test_extract_features_peak_freq(self):
        example = np.array([0, -5, 6, 16])
        features = FourierExtractor().extract_features(example)
        # Expected value: 1000 Hz
        self.assertTrue(np.abs(features["peak-freq"] - 1000) < 0.001)

    def test_extract_features_central_freq(self):
        example = np.array([0, -5, 6, 16])
        features = FourierExtractor().extract_features(example)
        # Expected value: 562.31 Hz according to manual calc
        self.assertTrue(np.abs(features["central-freq"] - 562.3106514) < 0.001)

