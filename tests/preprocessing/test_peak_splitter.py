from unittest import TestCase
import numpy as np

from damage_identification.preprocessing.peak_splitter import PeakSplitter


class TestTwoPeakDetection(TestCase):
    def test_two_peaks(self):
        example = np.array(
            [
                0,
                0,
                0,
                1,
                2,
                2,
                8,
                15,
                90,
                35,
                2,
                2,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                2,
                3,
                3,
                4,
                4,
                5,
                6,
                40,
                36,
                56,
                90,
                50,
                6,
                2,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        self.peakdetection = PeakSplitter(example, 10, 2, 1, 2, 10)
        result = self.peakdetection.split_single()
        self.assertEqual(len(result), 2)

    def test_one_peak(self):
        example = np.array(
            [
                1,
                2,
                80,
                35,
                2,
                1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        self.peakdetection = PeakSplitter(example, 20, 2, 1, 5, 10)
        result = self.peakdetection.split_single()
        self.assertEqual(len(result), 1)
