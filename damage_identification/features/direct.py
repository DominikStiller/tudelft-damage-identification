from typing import Dict, Any, Optional

import numpy as np
from scipy.integrate import simpson

from damage_identification.features.base import FeatureExtractor
from damage_identification.io import load_uncompressed_data

waveform = load_uncompressed_data("data/1column.csv")

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
        - first_n_samples: baseline to compare other features
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
            params["direct_features_threshold"] = 0.5
        if "direct_features_n_samples" not in params:
            params["direct_features_n_samples"] = 6

        super().__init__("direct", params)

    def extract_features(self, example: np.ndarray) -> Dict[str, float]:
        """
        Extracts direct features from a single waveform.

        Args:
            example: a single example (shape 1 x length_example)

        Returns:
            A dictionary containing items with each feature name value for the input example.
        """
        # counts, only upwards positive crossings of threshold
        example = example.flatten()
        n_samples = len(example)

        threshold = self.params["direct_features_threshold"]
        n_sample = min(self.params["direct_features_n_samples"], n_samples)
        above_threshold = np.abs(example) >= abs(threshold)
        count = 0
        i_array = np.array((0, 0))
        for i in range(len(above_threshold)):
            if above_threshold[i] and not above_threshold[i - 1]:
                count = count + 1
                if count == 1:
                    i_array[0] = i
            if above_threshold[i]:
                i_array[-1] = i

        # duration
        duration = (i_array[-1] - i_array[0]) * 1 / n_samples / 1000  # in s!

        # peak amplitude
        peak_amplitude = np.max(np.abs(example))
        peak_amplitude_index = np.argmax(np.abs(example))

        # rise time
        rise_time = (peak_amplitude_index - i_array[0]) * 1 / n_samples / 1000  # in s

        # energy (squared micro-volt for 1/1000th second --> 10e-12V)
        time_stamps = np.linspace(0, 1 / 1000, n_samples)  # in s
        energy = simpson(np.square(example * 1000), time_stamps)

        return_dict = {
            "peak_amplitude": peak_amplitude,
            "count": count,
            "duration": duration,
            "rise_time": rise_time,
            "energy": energy,
        }

        # n-sample
        return_dict.update({"n_sample_" + str(n + 1): example[n] for n in range(n_sample)})

        # Testing for signal peak in signal:
        boundary_index = round(n_samples * 0.2)  # Boundary of first damage mode in signal
        cut_waveform_1 = example[:boundary_index]
        peakamplitude_1_index = np.argmax(np.abs(cut_waveform_1))
        cut_waveform_2 = example[boundary_index:]
        peakamplitude_2_index = np.argmax(np.abs(cut_waveform_2)) + boundary_index
        # Check if we have two peaks with 60% difference in the same signal
        if (
            abs(example[peakamplitude_2_index] - example[peakamplitude_1_index])
            / max(example[peakamplitude_2_index], example[peakamplitude_1_index])
            < 0.6
        ):
            # Setting any feature to None marks this example as invalid
            return_dict["duration"] = None
        return return_dict

results = DirectFeatureExtractor()
print(results.extract_features(waveform))