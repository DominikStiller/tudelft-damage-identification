from unittest import TestCase

import pandas as pd
from numpy import testing


import numpy as np

from damage_identification.pre_processing.wavelet_filtering import WaveletFiltering


class TestWaveletFiltering(TestCase):
    def test_wavelet_filtering(self):

        data = WaveletFiltering().load_data('../../data/Waveforms.csv')
        print(data.dtype)
        test = WaveletFiltering().filtering(data[1,:])
        print(test)
        testing.assert_array_equal(test, [1, 2, 3, 5, 6])
