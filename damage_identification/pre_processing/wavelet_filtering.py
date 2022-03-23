from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import spkit as sp

from damage_identification import io

from damage_identification.pre_processing.base import PreProcessing

matplotlib.rcParams["figure.dpi"] = 300


class WaveletFiltering(PreProcessing):
    """
    This class pre-processes the data to remove noise from the data, either with a manual wavelet coefficient or with a
    mathematically determined threshold.

    Parameters:
        - wavelet_family: the wavelet family name; either db for daubechies or coif for Coiflet
        - wavelet_scale: the magnification scale of the wavelet family from 3-38 for daubechies or 1-17 for coiflet
    """

    def __init__(self, params: Dict[str, Any]):
        """
        This method initializes the waveform class and sets the parameters for the filtering  method to preprocess the
        data.

        Args:
            params: parameters for the filtering method

        """
        super(WaveletFiltering, self).__init__("prep", params)
        self.wavelet_fam = self.params["wavelet_family"]
        self.wavelet_scale = self.params["wavelet_scale"]
        self.waveform = []
        self.prep_waveform = []

    def load_data(self, data):
        """
        Loads the waveform data from an external file into the object to preprocess and/or plot.

        Args:
            data: the data file name of the waveform data to be processed.
        """
        self.waveform = io.load_uncompressed_data(data)
        return self.waveform

    def filtering(self, data, threshold: any):
        """
        Filters the waveform data based on the object filtering parameters set initially, and the noise threshold set as
        either an optimisation method or a numeric value.

        Args:
            data: waveform signal data file name.
            threshold: Either a numeric value or a threshold optimisation method; optimal or iqr or sd
        """
        x = data

        if self.wavelet_fam != 'db' and self.wavelet_fam != 'coif':
            raise ValueError('Not a valid wavelet family.')
        elif (17 >= self.wavelet_scale >= 3) or (3 > self.wavelet_scale >= 1 and self.wavelet_fam == 'coif') or (38 >= self.wavelet_scale > 17 and self.wavelet_fam == 'db'):
            if type(threshold) != float and int:
                if threshold == 'optimal' or 'iqr' or 'sd':
                    xf = sp.wavelet_filtering(x.copy(), wv=self.wavelet_fam+str(self.wavelet_scale), threshold=threshold, verbose=0, WPD=False)
                    return xf
                else:
                    raise ValueError('Not a valid threshold method.')
            else:
                xf = sp.wavelet_filtering(x.copy(), wv=self.wavelet_fam+str(self.wavelet_scale), threshold=threshold, verbose=0, WPD=False)
                return xf
        else:
            raise ValueError('Not a valid wavelet, scale configuration')

    def wavelet_plot(self):
        """
        Plots the first signal of the (preprocessed) waveform.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.waveform[0,:], label='Raw', color='b')
        plt.plot(self.prep_waveform[0,:], label='Filtered', color='r')
        plt.legend()
        # plt.title(f"DWT Denoising with {self.wave_fam+str(self.wave_scale)} Wavelet", size=15)
        return plt.show()

    def prep(self, threshold: any):
        """
        Goes through all waveform rows to process all data rows.

        Args:
            threshold: noise threshold level as numeric value or threshold method; optimal or iqr or sd
        """
        i = 0
        while i < len(self.waveform):
            self.prep_waveform.append(self.filtering((self.waveform[i, :]), threshold))
            i += 1
        self.prep_waveform = np.array(self.prep_waveform)
        return self.prep_waveform
