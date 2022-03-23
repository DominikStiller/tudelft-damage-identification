from unittest import TestCase
from damage_identification.pre_processing.wavelet_filtering import WaveletFiltering


class TestWaveletFiltering(TestCase):
    def test_wavelet_filtering(self):
        wave_object = WaveletFiltering({"wavelet_family": "db", "wavelet_scale": 3})
        wave_object.load_data("data/Waveforms.csv")
        wave_object.filtering(wave_object.waveform, "optimal")

    def test_signal_prep(self):
        wave = WaveletFiltering({"wavelet_family": "db", "wavelet_scale": 3})
        wave.waveform = wave.load_data("data/Waveforms.csv")
        waves = wave.prep("optimal")
        wave.wavelet_plot(1)

    def test_testing(self):
        wave = WaveletFiltering({"wavelet_family": "db", "wavelet_scale": 3})
        wave.load_data("data/Wavelet_validate.csv")
        waves = wave.prep("optimal")
        wave.wavelet_plot(0)
