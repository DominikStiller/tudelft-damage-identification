from typing import Any, Optional

import numpy as np
from scipy.integrate import simpson

from damage_identification.features.base import FeatureExtractor


class DirectFeatureExtractor(FeatureExtractor):
    """
    This class extracts all features that can be obtained directly from the waveform without further transformation.

    List of features:
        - peak_amplitude: maximum absolute value of the waveform
        - counts: number of upwards crossings of the threshold
        - duration: length in time
        !- rise_time: time required to increase from one specified value (e.g. 10% amplitude) to another
        (e.g. 90% amplitude)
        - energy: the energy of certain frequency bands in different section of the waveform
        - sample_X: first n samples as baseline to compare other features
    """

    def __init__(self, params: Optional[dict[str, Any]] = None):
        """
        Initialize the direct feature extractor.

        Args:
            params: parameters for the feature extractor, uses default parameters if None
        """
        if params is None:
            params = {}
        if "direct_features_threshold" not in params:
            params["direct_features_threshold"] = 0.4
        if "direct_features_n_samples" not in params:
            params["direct_features_n_samples"] = 0
        if "max_relative_peak_amplitude" not in params:
            params["max_relative_peak_amplitude"] = 0.5
        if "first_peak_domain" not in params:
            params["first_peak_domain"] = 0.2
        super().__init__("direct", params)

    def extract_features(self, example: np.ndarray) -> dict[str, float]:
        """
        Extracts direct features from a single waveform.

        Args:
            example: a single example (shape 1 x n_samples)

        Returns:
            A dictionary containing items with each feature name value for the input example.
        """
        example = example.flatten()
        n_samples = len(example)
        sampling_rate = self.params["sampling_rate"]  # n_samples per millisecond

        threshold = self.params["direct_features_threshold"]
        first_peak_domain = self.params["first_peak_domain"]

        assert 0 < threshold < 1, "Threshold must be between 0 and 1"
        assert 0 < first_peak_domain < 1, "First peak domain boundary must be between 0 and 1"

        # Peak amplitude
        peak_amplitude = np.max(np.abs(example))
        peak_amplitude_index = np.argmax(np.abs(example))

        above_threshold = (np.abs(example) >= abs(threshold * peak_amplitude)).astype(int)
        diffs = above_threshold[1:] - above_threshold[:-1]

        # Only count inner to outer crossings
        #    (i.e. positive crossings on positive side, negative crossing on negative side)
        counts = np.sum(diffs == 1)

        # Duration
        if diffs.any():
            # Only calculate duration if there is at least one sample above threshold
            duration_start_index = 1 + np.nonzero(diffs)[0][0]
            duration_end_index = np.argwhere(above_threshold)[-1][0]
            duration = (duration_end_index - duration_start_index) / sampling_rate

            # Rise time
            rise_time = (peak_amplitude_index - duration_start_index) / sampling_rate  # in s
        else:
            duration = 0
            rise_time = 0

        # Energy
        # AE values are in μV -> energy in this calculation is aJ (e-18)
        timestamps = np.linspace(0, 1 / 1000, n_samples)  # in s
        energy = simpson(np.square(example * 1000), timestamps)

        # First n samples
        first_n_samples = {
            "sample_" + str(n + 1): example[n]
            for n in range(min(self.params["direct_features_n_samples"], n_samples))
        }

        return {
            "peak_amplitude": peak_amplitude,
            "counts": counts,
            "duration": duration,
            "rise_time": rise_time,
            "energy": energy,
        } | first_n_samples
