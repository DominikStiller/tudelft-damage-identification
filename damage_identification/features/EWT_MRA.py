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
    def __init__(self, wavelet, mode):
        """
        Description
        """
        self.signal_data = []
        self.decomposed_data = np.ndarray([])
        self.reconstructed = np.ndarray([])
        self.wavelet = wavelet
        self.mode = mode
        # self.mfb = np.ndarray([])
        # self.boundaries = np.ndarray([])
        # super(MultiResolutionAnalysis, self).__init__("EWT_MRA", params)

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

    def wpt_mra(self):
        """
        Description
        """
        wp = pywt.WaveletPacket(data=self.signal_data, wavelet=self.wavelet, mode=self.mode)
        return

    # def plot_decomposition(self):
    #     plt.plot(self.decomposed_data)
    #     plt.show()
    #     _, decomp_lvl = np.shape(self.decomposed_data)
    #     for i in range(self.decomposed_data.shape[1]):
    #         plt.subplot(decomp_lvl, 1, i + 1)
    #         plt.plot(self.decomposed_data[:, i])
    #     return plt.show()

    def data_handler(self):
        coeffs = []
        wp = pywt.WaveletPacket(data=self.signal_data, wavelet=self.wavelet, mode=self.mode)
        print(wp.maxlevel, self.signal_data)
        for level in range(1,wp.maxlevel+1):
            print(level)
            coeffs.append([wp[node.path].data for node in wp.get_level(level, 'natural')])

        # energy_lvl1 = sum((coeffs[0][0][:])**2)
        energy_lvl1 = sum((coeffs[0][0][:])) + sum((coeffs[0][1][:]))
        energy_lvl3 = sum((coeffs[2][0][:]) ** 2) + sum((coeffs[2][1][:]) ** 2) + sum((coeffs[2][2][:]) ** 2) + sum((coeffs[2][3][:]) ** 2)
        energy_lvl3_1 = sum((coeffs[2][0][:]) ** 2) / energy_lvl3
        energy_lvl3_2 = sum((coeffs[2][1][:]) ** 2) / energy_lvl3
        energy_lvl3_3 = sum((coeffs[2][2][:]) ** 2) / energy_lvl3
        energy_lvl3_4 = sum((coeffs[2][3][:]) ** 2) / energy_lvl3
        # print(energy_lvl1, energy_lvl2, energy_lvl3)

        plt.plot(np.linspace(0, len(coeffs[0][1][:]), len(coeffs[0][1][:])), coeffs[0][1][:], color="g")
        plt.plot(np.linspace(0, len(self.signal_data), len(self.signal_data)), self.signal_data)
        # plt.plot(np.linspace(0, len(coeffs[2][0][:]),len(coeffs[2][0][:])), coeffs[2][0][:], color="b")
        # plt.plot(np.linspace(541, 541+len(coeffs[2][1][:]),len(coeffs[2][1][:])), coeffs[2][1][:], color="r")

        plt.show()

        return print(coeffs[0][0], coeffs[0][1], len(coeffs[0][0]), energy_lvl3_1, energy_lvl3_2, energy_lvl3_3, energy_lvl3_4)

    def constructor(self, coeffs):
        pywt.waverec(coeffs, 'db1')
        return

    # def iewt1d(self, ewt, mfb):
    #     real = all(np.isreal(ewt[:,0]))
    #     _, decomp_lvl = np.shape(self.decomposed_data)
    #     if real:
    #         self.reconstructed = np.zeros(len(ewt[:, 0]))
    #         for i in range(0, decomp_lvl-1):
    #             self.reconstructed += np.real(np.fft.ifft(np.fft.fft(ewt[:, i]) * mfb[::2, i]))
    #     else:
    #         self.reconstructed = np.zeros(len(ewt[:, 0])) * 0j
    #         for i in range(0, decomp_lvl-1):
    #             self.reconstructed += np.fft.ifft(np.fft.fft(ewt[:, i]) * mfb[::2, i])
    #     return

    def plot_recon(self):
        plt.plot(self.reconstructed)
        plt.show()


    def total_decompose(self):

        for i in range(len(self.signal_data)):
            self.decomposed_data, self.mfb, self.boundaries = ewtpy.EWT1D(self.signal_data[i+1, :])
