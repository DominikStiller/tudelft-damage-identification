import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import Dict, Any

import numpy as np

from base import FeatureExtractor

class Fourier(FeatureExtractor):
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

    def extract_features(self, example: np.ndarray) -> Dict[str, float]:
        N = np.size(example)
        ts = np.linspace(0, 1, N)
        trans = fft(example)
        Amplitude = np.abs(trans)
        Angle = np.angle(trans)
        print(Amplitude, Angle)

extractor = Fourier("Fourier", None)

# Reading csvs WILL BE REMOVED
WAV = pd.read_csv('Waveforms.csv', header=None)
wav_data = WAV.values

TIME = pd.read_csv('Time_hits.csv', header=None)
time_data = TIME.values

# testing 1 waveform
N_row = 100  # select 1 row for analysis
samples = WAV.iloc[:, N_row].to_numpy()
extractor.extract_features(samples)

