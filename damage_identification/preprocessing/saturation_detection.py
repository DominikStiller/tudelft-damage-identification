from typing import Any

import numpy as np


class SaturationDetection:
    """
    This class preprocesses the data by removing all signals for which the amplitude is greater than 0.0995,
    caused by the sensor being cut off at values greater than that due to saturation.
    """

    def __init__(self, params: dict[str, Any] = None):
        if "saturation_threshold" not in params:
            params["saturation_threshold"] = 0.0995
        self.params = params

    def filter_all(self, data):
        """
        Filter out all saturated waveforms
        """
        return data[np.where(np.amax(data, axis=1) <= self.params["saturation_threshold"])]
