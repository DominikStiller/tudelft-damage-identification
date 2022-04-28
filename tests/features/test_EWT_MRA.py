from unittest import TestCase
from damage_identification.features.EWT_MRA import MultiResolutionAnalysis
import numpy as np
import matplotlib.pyplot as plt


class TestMultiResolutionAnalysis(TestCase):
    def test_decomposition(self):
        mra = MultiResolutionAnalysis('db3', 'symmetric', 4, 3)
        mra.load('data/comp0.tradb', 11145)
        mra.decomposer(mra.data_handler())

    def test_short_signal_decomposition(self):
        mra = MultiResolutionAnalysis('db1', 'symmetric', 4, 3)
        mra.load_manual([1,2,3,4,-1,-2,-3,-4])
        mra.decomposer(mra.data_handler())