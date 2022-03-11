from typing import Dict, Any, Optional

import numpy as np
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
            params = {"feature_extractor_direct_threshold": 2.5e-2}
        super().__init__("direct", params)

    def extract_features(self, example: np.ndarray) -> Dict[str, float]:
        """
        Extracts direct features from a single waveform.

        Args:
            example: a single example (shape length_example)

        Returns:
            A dictionary containing items with each feature name value for the input example.
        """
        #counts
        threshold = self.params["feature_extractor_direct_threshold"]
        above_threshold = example >= threshold
        count = 0
        iarray = np.array((0,0))
        for i in range(len(above_threshold)):
            if above_threshold[i] and not above_threshold[i - 1]:
                count = count + 1
                if count ==1:
                    iarray[0] = i
            if above_threshold[i]:
                iarray[-1] = i
        #duration
        duration = (iarray[-1]-iarray[0])*1/2048 #in ms!
        #peak amplitude
        peak_amplitude = np.max(np.abs(example))
        print(iarray[:])
        return {"peak_amplitude": peak_amplitude, "count": count, "duration": duration}
        #rise time



results = DirectFeatureExtractor()


print(results.extract_features(waveform))
