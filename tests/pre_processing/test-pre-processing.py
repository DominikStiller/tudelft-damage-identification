from unittest import TestCase

import numpy as np
import sys

from damage_identification.preprocessing.wavelet_filtering import WaveletFiltering
from damage_identification.io import load_uncompressed_data, load_compressed_data


class TestWaveletFiltering(TestCase):
    # def test_wavelet_filtering(self):
    #     wave_object = WaveletFiltering({"wavelet_family": "db", "wavelet_scale": 3})
    #     wave_object.load_data("data/Waveforms.csv")
    #     wave_object.filtering(wave_object.waveform, "optimal")

    def test_signal_prep(self):
        if __name__ == "__main__":
            filter = WaveletFiltering({"wavelet_family": "db", "wavelet_scale": 3})
            raw = load_compressed_data("data/comp0.tradb")
            filtered = filter.filter(raw)
            # np.savetxt("sample.csv", filtered[6722, :], delimiter=",")
            filter.wavelet_plot(raw, filtered, 6722)
