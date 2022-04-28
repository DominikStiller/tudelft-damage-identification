from typing import Any, Optional

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

    def __init__(self, params: Optional[dict[str, Any]] = None):
        if params is None:
            params = {}
        super().__init__("fourier", params)

    def extract_features(self, example: np.ndarray) -> dict[str, float]:
        """
        Uses Fourier transform to extract peak frequency and central frequency features

        Args:
            example: a single example (shape 1 x n_samples)
        Returns:
            dictionary containing peak frequency ("peak_frequency") and central frequency ("central_frequency")
        """
        n_samples = np.size(example)
        ft = fft(example)
        freqs = fftfreq(n_samples, d=1 / self.params["sampling_rate"])
        ft[freqs < 0] = 0
        amp = np.abs(ft) / n_samples
        peakfreq = freqs[np.argmax(amp)]
        avgfreq = np.average(freqs, weights=amp)

        return {"peak_frequency": peakfreq, "central_frequency": avgfreq}

    def transform(self, example: np.ndarray) -> np.ndarray:
        length_example = np.size(example)
        amp = np.abs(fft(example))
        freqs = fftfreq(length_example, d=1 / self.params["sampling_rate"])
        amp = amp[freqs > 0]
        freqs = freqs[freqs > 0]
        return np.vstack((freqs, amp))
