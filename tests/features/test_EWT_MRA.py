from unittest import TestCase

from numpy import testing

from damage_identification import io
from damage_identification.features.EWT_MRA import MultiResolutionAnalysis


class TestMultiResolutionAnalysis(TestCase):
    # def test_decomposition(self):
    #     mra = MultiResolutionAnalysis(
    #         {
    #             "wavelet_decomposition_family": "db",
    #             "wavelet_magnitude": 8,
    #             "decomposition_time_bands": 8,
    #             "decomposition_level": 4,
    #         }
    #     )
    #     data = io.load_compressed_data("data/comp0.tradb")[11145, :]
    #     print(mra.decompose(data)[0])

    def test_short_signal_decomposition(self):
        mra = MultiResolutionAnalysis(
            {
                "wavelet_decomposition_family": "db",
                "wavelet_magnitude": 1,
                "decomposition_time_bands": 4,
                "decomposition_level": 2,
            }
        )
        test_decomposition, total_energy = mra.decompose(
            [1, 2, 3, 4, -1, -2, -3, -4, 1, 2, 3, 4, -1, -2, -3, -4]
        )
        known_decomposition = {
            "Frequency Band 0.0 - 512.0 kHz": [
                0.20833333333333337,
                0.20833333333333337,
                0.20833333333333337,
                0.20833333333333337,
            ],
            "Frequency Band 512.0 - 1024.0 kHz": [
                0.033333333333333354,
                0.033333333333333354,
                0.033333333333333354,
                0.033333333333333354,
            ],
            "Frequency Band 1024.0 - 1536.0 kHz": [
                0.008333333333333331,
                0.008333333333333331,
                0.008333333333333331,
                0.008333333333333331,
            ],
            "Frequency Band 1536.0 - 2048.0 kHz": [
                1.0271626370065256e-34,
                1.0271626370065256e-34,
                1.0271626370065256e-34,
                1.0271626370065256e-34,
            ],
        }

        self.assertAlmostEqual(total_energy, 1, 1)
        testing.assert_array_equal(test_decomposition, known_decomposition)
