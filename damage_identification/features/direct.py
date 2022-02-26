from typing import Dict, Any, Optional

import numpy as np

from damage_identification.features.base import FeatureExtractor


class DirectFeatureExtractor(FeatureExtractor):
    """
    This class extracts all features that can be obtained directly from the waveform without further transformation.

    List of features:
        - peak_amplitude: maximum absolute value of the waveform
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the direct feature extractor.

        Args:
            params: parameters for the feature extractor, uses default parameters if None
        """
        super().__init__("direct", params)

    def extract_features(self, example: np.ndarray) -> Dict[str, float]:
        """
        Extracts direct features from a single waveform.

        Args:
            example: a single example (shape length_example)

        Returns:
            A dictionary containing items with each feature name value for the input example.
        """
        peak_amplitude = np.max(np.abs(example))
        return {"peak_amplitude": peak_amplitude}
