from unittest import TestCase
from damage_identification.features.EWT_MRA import MultiResolutionAnalysis


class TestMultiResolutionAnalysis(TestCase):
    def test_decomposition(self):
        mra = MultiResolutionAnalysis()
        mra.signal_data = mra.load('data/Waveforms.csv')
        mra.ewt_mra()
