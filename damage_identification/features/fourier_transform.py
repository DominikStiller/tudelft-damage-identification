import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import Dict, Any
from base import FeatureExtractor


class Fourier(FeatureExtractor):
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

    def extract_features(self, example: np.ndarray) -> Dict[str, float]:
        length_example = np.size(example)
        ft = fft(example)
        ftfreq = fftfreq(example)
        amp = np.abs(ft)/length_example
        ang = np.angle(ft)

        return amp


extractor = Fourier("Fourier", None)

# Reading csvs WILL BE REMOVED
WAV = pd.read_csv('Waveforms.csv', header=None)
wav_data = WAV.values

TIME = pd.read_csv('Time_hits.csv', header=None)
time_data = TIME.values

# testing 1 waveform
N_row = 100  # select 1 row for analysis
row = WAV.iloc[:, N_row].to_numpy()
tspace = np.linspace(0, 1, np.size(row))

length_example = np.size(row)
ft = fft(row)
ftfreq = fftfreq(length_example)
amp = np.abs(ft)/length_example
ang = np.angle(ft)
plt.plot(ftfreq, ft)


plt.show()
