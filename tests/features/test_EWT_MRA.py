from unittest import TestCase
from damage_identification.features.EWT_MRA import MultiResolutionAnalysis
import numpy as np
import matplotlib.pyplot as plt


class TestMultiResolutionAnalysis(TestCase):
    def test_decomposition(self):
        mra = MultiResolutionAnalysis('db20', 'symmetric')
        mra.load('data/comp0.tradb', 11145)
        mra.data_handler()
        # print(mra.signal_data)
        # mra.plot_decomposition()

    def test_reconstructor(self):
        mra = MultiResolutionAnalysis('db20', 'symmetric')
        mra.load('data/comp0.tradb', 11145)
        mra.constructor(mra.wpt_mra())

    def test_short_signal_decomposition(self):
        mra = MultiResolutionAnalysis('db1', 'symmetric')
        mra.load_manual([1,2,3,4,-1,-2,-3,-4])
        mra.data_handler()