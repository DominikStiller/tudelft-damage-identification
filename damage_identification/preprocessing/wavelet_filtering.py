from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import spkit as sp

from damage_identification.io import load_uncompressed_data


class WaveletFiltering:
    """
    This class pre-processes the data to remove noise from the data, either with a manual wavelet coefficient or with a
    mathematically determined threshold.

    Parameters:
        - wavelet_family: the wavelet family name; either db for daubechies or coif for Coiflet
        - wavelet_scale: the magnification scale of the wavelet family from 3-38 for daubechies or 1-17 for coiflet

    Example usage:
    filter = WaveletFiltering()
    raw = load_uncompressed_data("data/Waveforms.csv")
    filtered = filter.filter(raw)
    filter.wavelet_plot(raw, filtered, 1)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        This method initializes the waveform class and sets the parameters for the filtering  method to preprocess the
        data.

        Args:
            params: parameters for the filtering method

        """
        if params is None:
            self.params: Dict[str, Any] = {}
        else:
            self.params = params
        if "wavelet_family" not in self.params:
            self.params["wavelet_family"] = "db"
        if "wavelet_scale" not in self.params:
            self.params["wavelet_scale"] = 3
        if "wavelet_threshold" not in self.params:
            self.params["wavelet_threshold"] = "optimal"

        # Check wavelet family
        self.wavelet_fam: str = self.params["wavelet_family"]
        if self.wavelet_fam not in ["db", "coif"]:
            raise Exception("Not a valid wavelet family.")

        # Check wavelet/scale combination
        self.wavelet_scale: int = self.params["wavelet_scale"]
        if not (
            (17 >= self.wavelet_scale >= 3)
            or (3 > self.wavelet_scale >= 1 and self.wavelet_fam == "coif")
            or (38 >= self.wavelet_scale > 17 and self.wavelet_fam == "db")
        ):
            raise Exception("Not a valid wavelet/scale configuration")

        # Check thresholding method
        if self.params["wavelet_threshold"] in ["optimal", "iqr", "sd"]:
            self.threshold = self.params["wavelet_threshold"]
        else:
            try:
                self.threshold = float(self.params["wavelet_threshold"])
            except ValueError:
                raise Exception("Not a valid threshold method for wavelet filtering")

    def _filter_single(self, data: np.ndarray) -> np.ndarray:
        """
        Filters the waveform data based on the object filtering parameters set initially, and the noise threshold set as
        either an optimisation method or a numeric value.

        Args:
            data: waveform signal data file name.
            threshold: Either a numeric value or a threshold optimisation method; optimal or iqr or sd
        """
        return sp.wavelet_filtering(
            data.copy(),
            wv=self.wavelet_fam + str(self.wavelet_scale),
            threshold=self.threshold,
            verbose=0,
            WPD=False,
        )

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Goes through all waveform rows to process all data rows.

        Args:
            threshold: noise threshold level as numeric value or threshold method; optimal or iqr or sd
        """
        return np.apply_along_axis(self._filter_single, 1, data)

    def wavelet_plot(self, raw, filtered, n):
        """
        Plots the first signal of the (preprocessed) waveform.

        Args:
            n: the column in the waveform data to plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(raw[n, :], label="Raw", color="b")
        plt.plot(filtered[n, :], label="Filtered", color="r")
        plt.legend()
        plt.title(f"DWT Denoising with {self.wavelet_fam+str(self.wavelet_scale)} Wavelet", size=15)
        plt.show()
