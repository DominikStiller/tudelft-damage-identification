from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import spkit as sp

from damage_identification import io

from damage_identification.pre_processing.base import PreProcessing

matplotlib.rcParams["figure.dpi"] = 400


class WaveletFiltering(PreProcessing):
    """
    This class pre-processes the data to remove noise from the data, either with a manual wavelet coefficient or with a
    mathematically determined threshold.

    Parameters:
        -
    """

    def __init__(self, params: Dict[str, Any]):
    # def __init__(self):
        """
        Description
        """
        super(WaveletFiltering, self).__init__("prep", params)
        self.wavelet_fam = self.params["wavelet_family"]
        self.waveform = []
        self.prep_waveform = []

    def load_data(self, data):
        """
        Description
        """
        self.waveform = io.load_uncompressed_data(data)
        return self.waveform

    def filtering(self, data, wave_scale: int, threshold: str):
        """
        Description
        """
        x = data

        if self.wavelet_fam != 'db' and self.wavelet_fam != 'coif':
            raise ValueError('Not a valid wavelet family.')
        elif (17 >= wave_scale >= 3) or (3 > wave_scale >= 1 and self.wavelet_fam == 'coif') or (38 >= wave_scale > 17 and wave_fam == 'db'):
            if type(threshold) != float and int:
                if threshold == 'optimal' or 'iqr' or 'sd':
                    xf = sp.wavelet_filtering(x.copy(), wv=self.wavelet_fam+str(wave_scale), threshold=threshold, verbose=0, WPD=False)
                    return xf
                else:
                    raise ValueError('Not a valid threshold method.')
            else:
                xf = sp.wavelet_filtering(x.copy(), wv=self.wavelet_fam+str(wave_scale), threshold=threshold, verbose=0, WPD=False)
                return xf
        else:
            raise ValueError('Not a valid wavelet, scale configuration')

        # xf_db = sp.wavelet_filtering(x.copy(), wv='db3', threshold='optimal', verbose=1, WPD=False)
        # xf_coif = sp.wavelet_filtering(x.copy(), wv='coif4', threshold='optimal', verbose=1, WPD=False)
        # t = np.arange(len(x))
        # t1 = np.arange(len(xf))
        # # t2 = np.arange(len(xf_coif))
        # plt.plot(t, x, 'r')
        # plt.plot(t1, xf, 'g')
        # # plt.plot(t2, xf_coif, 'b')
        # plt.show()

    def wavelet_plot(self, x, xf):
        plt.figure(figsize=(10, 6))
        plt.plot(x, label='Raw', color='b')
        plt.plot(xf, label='Filtered', color='r')
        plt.legend()
        # plt.title(f"DWT Denoising with {self.wave_fam+str(self.wave_scale)} Wavelet", size=15)
        return plt.show()

    def prep(self, wave_scale: int, threshold: str):
        i = 0
        while i < len(self.waveform):
            self.prep_waveform.append(self.filtering((self.waveform[i, :]), wave_scale, threshold))
            i += 1
        self.prep_waveform = np.array(self.prep_waveform)
        return self.prep_waveform


