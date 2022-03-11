import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fourier import FourierExtractor


'''
Testing 1 selected waveform and plotting it.
Reading csv's is temporary until the I/O is set up.

N_row selects the N+1th example from the WAV dataset.

'''


WAV = pd.read_csv("Waveforms.csv", header=None)
wav_data = WAV.values
N_row = 3000  # select row N+1 for analysis
row = WAV.iloc[:, N_row].to_numpy()

Fextractor = FourierExtractor()

features = Fextractor.extract_features(row)

print(f"Peak frequency = {features['peak-freq']} Hz\nCentral frequency = {features['central-freq']}")

plt.plot(features["ft-freq"], features["amp"])
plt.vlines(features["peak-freq"], 0, np.max(features["amp"]), linestyles="dotted", colors="r")
plt.vlines(features["central-freq"], 0, np.max(features["amp"]), linestyles="dashed", colors="g")

print(WAV.shape)
test_features = np.empty((0, 3))
for N in range(1, len(WAV.index)):
    row = WAV.iloc[:, N].to_numpy()
    features = Fextractor.extract_features(row)
    np.append(test_features, np.array([N, features["peak-freq"], features["central-freq"]]))
    #  print(N)

print(test_features)

plt.show()
