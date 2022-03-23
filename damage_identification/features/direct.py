from typing import Dict, Any, Optional

import numpy as np
from scipy.integrate import simpson

from damage_identification.features.base import FeatureExtractor
from damage_identification.io import load_uncompressed_data


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

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the direct feature extractor.

        Args:
            params: parameters for the feature extractor, uses default parameters if None
        """
        if params is None:
            params = {}
        if "direct_features_threshold" not in params:
            params["direct_features_threshold"] = 0.02
        if "direct_features_n_samples" not in params:
            params["direct_features_n_samples"] = 200
        if "direct_features_max_relative_peak_error" not in params:
            params["direct_features_max_relative_peak_error"] = 0.6
        if "direct_features_first_peak_domain" not in params:
            params["direct_features_first_peak_domain"] = 0.2
        super().__init__("direct", params)

    def extract_features(self, example: np.ndarray) -> Dict[str, float]:
        """
        Extracts direct features from a single waveform.

        Args:
            example: a single example (shape 1 x length_example)

        Returns:
            A dictionary containing items with each feature name value for the input example.
        """
        # counts, crossing of the threshold both positive and negative
        example = example.flatten()
        n_samples = len(example)
        sampling_rate = n_samples * 1000

        threshold = self.params["direct_features_threshold"]
        max_relative_peak_error = self.params["direct_features_max_relative_peak_error"]
        first_peak_domain = self.params["direct_features_first_peak_domain"]
        n_sample = min(self.params["direct_features_n_samples"], n_samples)

        assert 0 < first_peak_domain < 1, "First peak domain boundary must be between 0 and 1"

        # peak amplitude
        peak_amplitude = np.max(np.abs(example))
        peak_amplitude_index = np.argmax(np.abs(example))

        above_threshold = (np.abs(example) >= abs(threshold)).astype(int)
        diffs = above_threshold[1:] - above_threshold[:-1]

        # Only count inner to outer crossings
        #    (i.e. positive crossings on positive side, negative crossing on negative side)
        counts = np.sum(diffs == 1)

        # duration
        if diffs.any():
            # Only calculate duration if there is at least one sample above threshold
            duration_start_index = 1 + np.nonzero(diffs)[0][0]
            duration_end_index = np.argwhere(above_threshold)[-1][0]
            duration = (duration_end_index - duration_start_index) / sampling_rate

            # rise time
            rise_time = (peak_amplitude_index - duration_start_index) / sampling_rate  # in s
        else:
            duration = 0
            rise_time = 0

        # energy (squared micro-volt for 1/1000th second --> 10e-12V)
        time_stamps = np.linspace(0, 1 / 1000, n_samples)  # in s
        energy = simpson(np.square(example * 1000), time_stamps)

        return_dict = {
            "peak_amplitude": peak_amplitude,
            "counts": counts,
            "duration": duration,
            "rise_time": rise_time,
            "energy": energy,
        }

        # n-sample
        return_dict.update({"sample_" + str(n + 1): example[n] for n in range(n_sample)})

        # Testing for signal peaks in signal:
        boundary_index = round(
            n_samples * first_peak_domain
        )  # Boundary of first damage mode in signal
        cut_waveform_1 = example[:boundary_index]
        peakamplitude_1 = np.max(np.abs(cut_waveform_1))
        cut_waveform_2 = example[boundary_index:]
        peakamplitude_2 = np.max(np.abs(cut_waveform_2)) + boundary_index
        relative_peak_error = abs(peakamplitude_2 - peakamplitude_1) / max(
            peakamplitude_2, peakamplitude_1
        )

        # Check if we have two peaks with max_relative_peak_error difference (60% by default) in the same signal
        if relative_peak_error < max_relative_peak_error:
            # Setting any feature to None marks this example as invalid
            return_dict["duration"] = None
        return return_dict


DirectFeatureExtractor().extract_features(load_uncompressed_data("data/Waveforms.csv")[0, :])
