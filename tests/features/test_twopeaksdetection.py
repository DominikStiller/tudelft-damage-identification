from unittest import TestCase
import numpy as np

from damage_identification.features.multiplepeakdetection import real_time_peak_detection


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
        self.peakdetection = real_time_peak_detection(example, 10, 2, 1)
        result = self.peakdetection.thresholding_algo()
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), len(example))
        self.assertEqual(len(result[1]), len(example))

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
        self.peakdetection = real_time_peak_detection(example, 20, 2, 1, 5)
        result = self.peakdetection.thresholding_algo()
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), len(example))
        print(result[1])
        print("/n")
        print(result[0])
        self.assertEqual(result[1], None)
