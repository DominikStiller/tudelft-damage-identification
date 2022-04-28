from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import ewtpy
import pywt
from damage_identification import io

matplotlib.rcParams['figure.dpi'] = 300

# T = 1000
# t = np.arange(1,T+1)/T
# f = np.cos(2*np.pi*0.8*t) + 2*np.cos(2*np.pi*10*t)+0.8*np.cos(2*np.pi*100*t)
# ewt,  mfb ,boundaries = ewtpy.EWT1D(f, N = 3)
# plt.plot(f)
# plt.plot(ewt)
# plt.show()


class MultiResolutionAnalysis:
    """
    Description
    """

    # def __init__(self, params: Dict[str: Any]):
    def __init__(self, wavelet, mode, time_bands, dec_level):
        """
        Description
        """
        self.signal_data = []
        self.wavelet = wavelet
        self.mode = mode
        self.time_bands = time_bands
        self.dec_level = dec_level

    def load(self, directory,n):
        """
        Description
        """
        x = io.load_compressed_data(directory)
        self.signal_data = x[n, :]
        return self.signal_data

    def load_manual(self, data):
        self.signal_data = data
        return self.signal_data

    def data_handler(self):
        coeffs = []
        wp = pywt.WaveletPacket(data=self.signal_data, wavelet=self.wavelet, mode=self.mode)
        # print(wp.maxlevel, self.signal_data)
        for level in range(1,wp.maxlevel+1):
            # print(level)
            coeffs.append([wp[node.path].data for node in wp.get_level(level, 'natural')])

        # energy_lvl1 = sum((coeffs[0][0][:])**2)
        # energy_lvl1 = sum((coeffs[0][0][:])**2) + sum((coeffs[0][1][:])**2)
        # energy_lvl3 = sum((coeffs[1][0][:]) ** 2) + sum((coeffs[1][1][:]) ** 2) + sum((coeffs[1][2][:]) ** 2) + sum((coeffs[1][3][:]) ** 2)
        # energy_lvl3_1 = sum((coeffs[1][0][:]) ** 2) / energy_lvl3
        # energy_lvl3_2 = sum((coeffs[1][1][:]) ** 2) / energy_lvl3
        # energy_lvl3_3 = sum((coeffs[1][2][:]) ** 2) / energy_lvl3
        # energy_lvl3_4 = sum((coeffs[1][3][:]) ** 2) / energy_lvl3
        # print(energy_lvl1, energy_lvl2, energy_lvl3)

        # plt.plot(np.linspace(0, len(coeffs[0][1][:]), len(coeffs[0][1][:])), coeffs[0][1][:], color="g")
        # plt.plot(np.linspace(0, len(self.signal_data), len(self.signal_data)), self.signal_data)
        # plt.plot(np.linspace(0, len(coeffs[2][0][:]),len(coeffs[2][0][:])), coeffs[2][0][:], color="b")
        # plt.plot(np.linspace(541, 541+len(coeffs[2][1][:]),len(coeffs[2][1][:])), coeffs[2][1][:], color="r")

        # plt.show()

        return coeffs

    def decomposer(self, coeffs):
        # print(coeffs)
        energy = []
        tot_energy = sum(coeffs[0][0][:]**2)+sum(coeffs[0][1][:]**2)
        for i in range(0,2**self.dec_level):
            # print(len(coeffs[self.dec_level-1][i]))
            # print(coeffs[self.dec_level-1][i][:])
            energy.append([sum(coeffs[self.dec_level-1][i][:]**2) / tot_energy])
            for t in range(0, len(coeffs[self.dec_level-1][i][:])):
                # print(t)
                energy.append([sum(coeffs[(self.dec_level-1)][i][:]**2)/tot_energy])

        return print(energy)
