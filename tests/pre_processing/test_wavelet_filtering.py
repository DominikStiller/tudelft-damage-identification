from unittest import TestCase
import tempfile
import os
import pandas as pd
from numpy import testing


import numpy as np

from damage_identification.pre_processing.wavelet_filtering import WaveletFiltering


class TestWaveletFiltering(TestCase):
    def test_wavelet_filtering(self):
        data = WaveletFiltering().load_data('data/Waveforms.csv')
        WaveletFiltering().filtering(data[0, :], 'db', 5, 'optimal')

    def test_signal_prep(self):
        wave = WaveletFiltering()
        wave.waveform = wave.load_data('data/Waveforms.csv')
        waves = wave.prep( 5, 'optimal')
        # wave.wavelet_plot(wave.waveform[1,:], waves[1,:])

    def test_sine(self):
        wave = WaveletFiltering()
        wave.waveform = np.transpose(np.array([np.sin(2*np.pi*np.linspace(0, 1, 1000))]))
        print(wave.waveform)
        waves = wave.prep( 10, 'optimal')
        wave.wavelet_plot(wave.waveform, waves)