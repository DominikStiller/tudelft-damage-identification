from unittest import TestCase

import numpy as np
from numpy import testing

from damage_identification.features.mra import MultiResolutionAnalysisExtractor


class TestMultiResolutionAnalysis(TestCase):
    # def test_decomposition(self):
    #     mra = MultiResolutionAnalysis(
    #         {
    #             "mra_wavelet_family": "db",
    #             "mra_wavelet_scale": 8,
    #             "mra_time_bands": 8,
    #             "mra_levels": 4,
    #         }
    #     )
    #     data = io.load_compressed_data("data/comp0.tradb")[11145, :]
    #     print(mra.extract_features(data))

    def test_short_signal_decomposition(self):
        mra = MultiResolutionAnalysisExtractor(
            {
                "mra_wavelet_family": "db",
                "mra_wavelet_scale": 1,
                "mra_time_bands": 4,
                "mra_levels": 2,
            }
        )
        test_decomposition = mra.extract_features(
            np.array([1, 2, 3, 4, -1, -2, -3, -4, 1, 2, 3, 4, -1, -2, -3, -4])
        )
        known_decomposition = {
            "mra_0_512_0": 0.20833333333333337,
            "mra_0_512_1": 0.20833333333333337,
            "mra_0_512_2": 0.20833333333333337,
            "mra_0_512_3": 0.20833333333333337,
            "mra_512_1024_0": 0.033333333333333354,
            "mra_512_1024_1": 0.033333333333333354,
            "mra_512_1024_2": 0.033333333333333354,
            "mra_512_1024_3": 0.033333333333333354,
        }
        testing.assert_array_equal(test_decomposition, known_decomposition)
