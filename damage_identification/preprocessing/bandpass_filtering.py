from typing import Any, Optional

import numpy as np
from scipy import signal


class BandpassFiltering:
    def __init__(self, params: Optional[dict[str, Any]] = None):
        if params is None:
            self.params: dict[str, Any] = {}
        else:
            self.params = params
        if "bandpass_low" not in self.params:
            self.params["bandpass_low"] = 100
        if "bandpass_high" not in self.params:
            self.params["bandpass_high"] = 900
        if "bandpass_order" not in self.params:
            self.params["bandpass_order"] = 25

        self.bandpass_filter = signal.butter(
            self.params["bandpass_order"],
            [self.params["bandpass_low"] * 1e3, self.params["bandpass_high"] * 1e3],
            "bandpass",
            fs=self.params["sampling_rate"],
            output="sos",
        )

    def filter_single(self, data: np.ndarray) -> np.ndarray:
        """
        Filters the waveform data using a Butterworth bandpass filter.

        Args:
            data: waveform signal data
        """
        return signal.sosfilt(self.bandpass_filter, data)

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Goes through all waveform rows to process all data rows.

        Args:
            data: waveform signal data
        """
        return np.apply_along_axis(self.filter_single, axis=1, arr=data)
