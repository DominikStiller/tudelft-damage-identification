import os
import pickle
from typing import Dict, Any
import numpy as np
import pandas as pd
from damage_identification import io


class WaveletFiltering:
    """
    This class pre-processes the data to remove noise from the data, either with a manual wavelet coefficient or with a
    mathematically determined threshold.

    Parameters:
        -
    """

    def __init__(self):
        """
        Description
        """
        # self.waveform = io.load_uncompressed_data('Waveforms.csv')
        self.waveform = []

    def load_data(self, data):
        self.waveform = io.load_uncompressed_data(data)


    def filtering(self, data):
        """
        Description
        """
        # x = self.waveform[:, 1]
        # t = np.arange(len(x))
        self.waveform = np.append(data, [5, 6])
        return self.waveform
