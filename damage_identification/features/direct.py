from typing import Dict

import numpy as np

from damage_identification.features.base import FeatureExtractor


class DirectFeatureExtractor(FeatureExtractor):
    """
    This class extracts all features that can be obtained directly from the waveform without further transformation.

    List of features:
        - Peak amplitude: maximum absolute value of the waveform
    """

    def get_name(self) -> str:
        return "direct"

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
