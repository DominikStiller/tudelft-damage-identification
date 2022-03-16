import os
import pickle
from typing import Dict, Any
import numpy as np
import pandas as pd
from damage_identification import io
import matplotlib.pyplot as plt
import matplotlib
import spkit as sp

matplotlib.rcParams["figure.dpi"] = 500


class WaveletFiltering:
    """
    This class pre-processes the data to remove noise from the data, either with a manual wavelet coefficient or with a
    mathematically determined threshold.

    Parameters:
        -
    """

    # def __init__(self, params: Dict[str, Any]):
    def __init__(self):
        """
        Description
        """
        # self.waveform = io.load_uncompressed_data('Waveforms.csv')
        self.waveform = []
        # super(WaveletFiltering, self).__init__("waveletfiltering", params)

    def load_data(self, data):
        self.waveform = io.load_uncompressed_data(data)
        return self.waveform

    def filtering(self, data, wave_fam, threshold):
        """
        Description
        """
        x = data
        if wave_fam != 'db4' and 'coif4':
            raise ValueError('Not a valid Wavelet.')
        else:
            if type(threshold) != float and int:
                if threshold == 'optimal' or 'iqr' or 'sd':
                    xf = sp.wavelet_filtering(x.copy(), wv=wave_fam, threshold=threshold, verbose=1, WPD=False)
                else:
                    raise ValueError('Not a valid threshold method.')
            else:
                xf = sp.wavelet_filtering(x.copy(), wv=wave_fam, threshold=threshold, verbose=1, WPD=False)

        # xf_db = sp.wavelet_filtering(x.copy(), wv='db3', threshold='optimal', verbose=1, WPD=False)
        # xf_coif = sp.wavelet_filtering(x.copy(), wv='coif4', threshold='optimal', verbose=1, WPD=False)
        # t = np.arange(len(x))
        # t1 = np.arange(len(xf_db))
        # t2 = np.arange(len(xf_coif))
        # plt.plot(t, x, 'r')
        # plt.plot(t1, xf_db, 'g')
        # plt.plot(t2, xf_coif, 'b')
        # plt.show()

        return xf
