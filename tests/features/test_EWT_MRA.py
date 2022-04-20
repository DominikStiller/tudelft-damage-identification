from unittest import TestCase
from damage_identification.features.EWT_MRA import MultiResolutionAnalysis
import numpy as np
import matplotlib.pyplot as plt


class TestMultiResolutionAnalysis(TestCase):
    def test_decomposition(self):
        mra = MultiResolutionAnalysis('db12', 'symmetric')
        mra.load('data/comp0.tradb', 11145)
        mra.wpt_mra()
        # print(mra.signal_data)
        # mra.plot_decomposition()

    def test_reconstructor(self):
        mra = MultiResolutionAnalysis()
        mra.load('data/Waveforms.csv')
        mra.wpt_mra()
        mfb = mra.decomposed_data
        print(mra.boundaries)
        print(mfb.shape)
        mra.iewt1d(mra.decomposed_data, mra.mfb)
        plt.plot(mra.signal_data[1, :])
        mra.plot_recon()
