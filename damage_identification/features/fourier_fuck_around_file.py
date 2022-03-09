import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import random as rnd
'''
from base import FeatureExtractor

class Fourier(FeatureExtractor):
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

'''

t = np.linspace(0, 4 * np.pi, 1000)
y = np.sin(t) + np.cos(3*t)

for element in range(len(y)):
    y[element] = y[element] + rnd.randint(1,1000)/1000 - np.sin(rnd.randint(1,10)/100)

ytrans = fft(y)
amp = np.abs(ytrans)
freq = fftfreq(1000)
amp = amp / max(amp)

plt.plot(t, y)
plt.show()
plt.plot(freq, amp)
plt.xlim(0,1)
plt.show()

filtered = ytrans.copy()
filtered[np.abs(filtered) < max(np.abs(filtered))/2] = 0
filtered[freq < 0] = 0
ampfilt = np.abs(filtered)
#ampfilt = ampfilt / max(ampfilt)

plt.plot(freq, ampfilt)
plt.title('filtered')
plt.show()

cleansig = ifft(filtered)

cleansig = np.real(cleansig)
plt.plot(t, cleansig)
plt.title('filteredwave')
plt.show()