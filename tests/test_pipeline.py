from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from damage_identification.features.base import FeatureExtractor
from damage_identification.pipeline import Pipeline


class TestPipeline(TestCase):
    def setUp(self):
        self.pipeline = Pipeline({})

        feature_extractor = Mock(spec=FeatureExtractor)
        feature_extractor.extract_features.side_effect = [
            {
                "feature_a": 1,
                "feature_b": 2,
            },
            {
                "feature_a": 3,
                "feature_b": 4,
            },
            {
                "feature_a": None,
                "feature_b": 5,
            },
        ]
        self.pipeline.feature_extractors = [feature_extractor]

    def test_extract_features(self):
        """Validate format of features DataFrame"""
        features, valid_mask = self.pipeline._extract_features(np.zeros((3, 2048)), 3)

        self.assertEqual(features.shape, (3, 2))
        self.assertListEqual(features.columns.tolist(), ["feature_a", "feature_b"])
        self.assertTrue(valid_mask[0])
        self.assertTrue(valid_mask[1])
        self.assertFalse(valid_mask[2])
