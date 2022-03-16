from typing import Dict, Any, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import pandas as pd

from damage_identification.features.base import FeatureExtractor
from damage_identification.io import load_uncompressed_data

example = load_uncompressed_data("1column.csv")
waveform = load_uncompressed_data("1column.csv")


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
            params = {"feature_extractor_direct_threshold": -2.5e-2, "feature_extractor_direct_n_sample": 6}

        super().__init__("direct", params)

    def extract_features(self, example: np.ndarray) -> Dict[str, float]:
        """
        Extracts direct features from a single waveform.

        Args:
            example: a single example (shape length_example)

        Returns:
            A dictionary containing items with each feature name value for the input example.
        """
        # counts
        threshold = self.params["feature_extractor_direct_threshold"]
        n_sample = self.params["feature_extractor_direct_n_sample"]
        if threshold>0:
            above_threshold = example >= threshold
        else:
            above_threshold = example <= threshold
        count = 0
        i_array = np.array((0,0))
        for i in range(len(above_threshold)):
            if above_threshold[i] and not above_threshold[i - 1]:
                count = count + 1
                if count == 1:
                    i_array[0] = i
            if above_threshold[i]:
                i_array[-1] = i

        # duration
        duration = (i_array[-1]-i_array[0])*1/2048/1000    # in m!

        # peak amplitude
        peak_amplitude = np.max(np.abs(example))
        peak_amplitude_index = np.argmax(example)

        # rise time
        rise_time = (peak_amplitude_index-i_array[0])*1/2048/1000  # in s

        # energy (squared micro-volt for 1/1000th second --> 10e-12V)
        time_stamps = np.linspace(0, 1/1000, len(example))       # in s
        cs = CubicSpline(time_stamps, example*1000)    # in micro-volt!
        f = lambda x: cs(x)**2
        print(f(time_stamps))
        energy = quad(f, time_stamps[0], time_stamps[-1], limit=100, epsabs=1e-10)

        # n-sample
        new_dict = {"n_sample_"+str(n+1): example[n] for n in range(n_sample)}
        return_dict = {"peak_amplitude": peak_amplitude, "count": count, "duration": duration, "rise_time": rise_time, "energy": energy[0]}
        final_dict = return_dict | new_dict

        #Testing for signal peak in signal:
        boundary_index = round(2048*0.2) #Boundary of first damage mode in signal
        cut_waveform_1 = example[: boundary_index]
        peakamplitude_1_index = np.argmax(np.abs(cut_waveform_1))
        cut_waveform_2 = example[boundary_index :]
        peakamplitude_2_index = np.argmax(np.abs(cut_waveform_2)) + boundary_index
        #Check if we have two peaks with 60% difference in the same signal
        if abs(example[peakamplitude_2_index] - example[peakamplitude_1_index])/max(example[peakamplitude_2_index], example[peakamplitude_1_index])<0.6:
            return None
        return final_dict


results = DirectFeatureExtractor()


print(results.extract_features(waveform))
