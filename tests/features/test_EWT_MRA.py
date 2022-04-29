from unittest import TestCase
from damage_identification.features.EWT_MRA import MultiResolutionAnalysis
from numpy import testing


class TestMultiResolutionAnalysis(TestCase):
    # def test_decomposition(self):
    #     mra = MultiResolutionAnalysis("db", 20, 8, 4)
    #     mra.load("data/comp0.tradb", 11145)
    #     mra.data_handler()
    #     print(mra.decomposer())

    def test_short_signal_decomposition(self):
        mra = MultiResolutionAnalysis("db", 1, 4, 2)
        mra.load_manual([1, 2, 3, 4, -1, -2, -3, -4, 1, 2, 3, 4, -1, -2, -3, -4])
        mra.data_handler()
        test_decomposition, total_energy = mra.decomposer()
        known_decomposition = [
            [0.20833333333333337, 0.20833333333333337, 0.20833333333333337, 0.20833333333333337],
            [
                0.033333333333333354,
                0.033333333333333354,
                0.033333333333333354,
                0.033333333333333354,
            ],
            [
                0.008333333333333331,
                0.008333333333333331,
                0.008333333333333331,
                0.008333333333333331,
            ],
            [
                1.0271626370065256e-34,
                1.0271626370065256e-34,
                1.0271626370065256e-34,
                1.0271626370065256e-34,
            ],
        ]
        self.assertAlmostEqual(total_energy, 1, 1)
        testing.assert_array_equal(test_decomposition, known_decomposition)
