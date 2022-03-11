import numpy as np
from scipy.fft import fft, fftfreq
from typing import Dict
from damage_identification.features.base import FeatureExtractor


class FourierExtractor(FeatureExtractor):
    """
    Feature extractor subclass using Fourier (scipy.fft) for a single example (waveform).

    init arguments: none needed

    The extract_features method returns a Dict with the following data:
    - Peak frequency as "peak-freq"
    - Central frequency as "central-freq"
    - Amplitude array as "amp" (for plotting)
    - Frequency domain array as "ft-freq" (for plotting)

        Typical usage example:
        Extractor = FourierExtractor()
        features = Extractor.extract_features(example) with example a np.ndarray
        peak_frequency = features['peak-freq']
    """
    def __init__(self):
        super().__init__("fourier", {})

    def extract_features(self, example: np.ndarray) -> Dict[str, float]:
        """
        Uses Fourier transform to extract peak frequency and central frequency features

        Args: 1-dimensional np.ndarray containing 1 example with (2048) samples of the waveform data
        Returns: dictionary containing peak frequency ("peak-freq"), central frequency ("central-freq")
        and amplitude np.ndarray ("amp") and the frequency domain ("ft-freq") for plotting.
        """

        length_example = np.size(example)
        ft = fft(example)
        freqs = fftfreq(length_example, d=0.001 / length_example)
        ft[freqs < 0] = 0
        amp = np.abs(ft) / length_example
        peakfreq = freqs[np.argmax(amp)]
        avgfreq = np.average(freqs, weights=amp)

        return {"peak-freq": peakfreq, "central-freq": avgfreq, "amp": amp, "ft-freq": freqs}
