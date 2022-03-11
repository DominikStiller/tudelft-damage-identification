from typing import Dict

import numpy as np
from scipy.fft import fft, fftfreq

from damage_identification.features.base import FeatureExtractor


class FourierExtractor(FeatureExtractor):
    """
    Feature extractor subclass using Fourier (scipy.fft) for a single example (waveform).

    init arguments: none needed

    The extract_features method returns a Dict with the following data:
    - Peak frequency as "peak_frequency"
    - Central frequency as "central_frequency"

    Typical usage example:
    Extractor = FourierExtractor()
    features = Extractor.extract_features(example) with example a np.ndarray
    peak_frequency = features["peak_frequency"]
    """

    def __init__(self):
        super().__init__("fourier", {})

    def extract_features(self, example: np.ndarray) -> Dict[str, float]:
        """
        Uses Fourier transform to extract peak frequency and central frequency features

        Args:
            example: a single example (shape 1 x length_example)
        Returns:
            dictionary containing peak frequency ("peak-freq") and central frequency ("central-freq")
        """

        length_example = np.size(example)
        ft = fft(example)
        freqs = fftfreq(length_example, d=0.001 / length_example)
        ft[freqs < 0] = 0
        amp = np.abs(ft) / length_example
        peakfreq = freqs[np.argmax(amp)]
        avgfreq = np.average(freqs, weights=amp)

        return {"peak_frequency": peakfreq, "central_frequency": avgfreq}
