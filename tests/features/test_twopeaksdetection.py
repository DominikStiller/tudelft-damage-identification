from unittest import TestCase
import numpy as np
from damage_identification.io import load_compressed_data

from damage_identification.features.multiplepeakdetection import real_time_peak_detection

pridbfile = load_compressed_data("data/comp0.tradb")
example1 = pridbfile[6722]
example2 = pridbfile[2]

class TestTwoPeakDetection(TestCase):


    def setUp(self):
        self.peakdetection = real_time_peak_detection(example1)

    def test_two_peaks(self):
        result = self.peakdetection.test_peak(example1)
        self.assertEqual(np.shape(result["firstslice"]), (2048,))
        self.assertEqual(np.shape(result["secondslice"]), (2048,))

    def test_one_peak(self):
        result = self.peakdetection.test_peak(example2)
        self.assertEqual(np.shape(result["firstslice"]), (2048,))
        self.assertEqual(result["secondslice"], None)