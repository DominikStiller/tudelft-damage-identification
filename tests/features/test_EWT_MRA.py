from unittest import TestCase
from damage_identification.features.EWT_MRA import MultiResolutionAnalysis
import numpy as np
import matplotlib.pyplot as plt


class TestMultiResolutionAnalysis(TestCase):
    def test_decomposition(self):
        mra = MultiResolutionAnalysis()
        mra.load("data/Waveforms.csv")
        mra.ewt_mra()
        # print(mra.signal_data)
        mra.plot_decomposition()

    def test_reconstructor(self):
        mra = MultiResolutionAnalysis()
        mra.load("data/Waveforms.csv")
        mra.ewt_mra()
        mfb = mra.decomposed_data
        print(mfb.shape)
        mra.iewt1d(mra.decomposed_data, mra.mfb)
        plt.plot(mra.signal_data[1, :])
        mra.plot_recon()
