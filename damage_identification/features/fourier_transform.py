import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from typing import Dict
from base import FeatureExtractor


class FourierExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__("fourier", {})

    def extract_features(self, example: np.ndarray) -> Dict[str, float]:
        length_example = np.size(example)
        ft = fft(example)
        print(ft)
        freqs = fftfreq(length_example, d=0.001 / 2048)
        ft[freqs < 0] = 0
        amp = np.abs(ft) / length_example
        peakfreq = freqs[np.argmax(amp)]
        avgfreq = np.average(freqs, weights=amp)

        return {"peak-freq": peakfreq, "central-freq": avgfreq, "amp": amp, "ft-freq": freqs}


# Reading csvs WILL BE REMOVED
# testing 1 waveform
WAV = pd.read_csv("Waveforms.csv", header=None)
wav_data = WAV.values
N_row = 100  # select 1 row for analysis
row = WAV.iloc[:, N_row].to_numpy()

Fextractor = FourierExtractor()

features = Fextractor.extract_features(row)

print(f"Peak frequency: {features['peak-freq']} Hz, {features['central-freq']}")

plt.plot(features["ft-freq"], features["amp"])
plt.vlines(features["peak-freq"], 0, np.max(features["amp"]), linestyles="dotted", colors="r")
plt.vlines(features["central-freq"], 0, np.max(features["amp"]), linestyles="dashed", colors="g")
plt.show()