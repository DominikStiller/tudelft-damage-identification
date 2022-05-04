from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import ewtpy
from damage_identification import io

matplotlib.rcParams["figure.dpi"] = 300

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
    def __init__(self):
        """
        Description
        """
        self.signal_data = []
        self.decomposed_data = np.ndarray([])
        self.reconstructed = np.ndarray([])
        self.mfb = np.ndarray([])
        self.boundaries = np.ndarray([])
        # super(MultiResolutionAnalysis, self).__init__("EWT_MRA", params)

    def load(self, directory):
        """
        Description
        """
        self.signal_data = io.load_uncompressed_data(directory)
        return self.signal_data

    def ewt_mra(self):
        """
        Description
        """
        self.decomposed_data, self.mfb, self.boundaries = ewtpy.EWT1D(self.signal_data[1, :])
        mfb = self.mfb
        print(mfb.shape)
        print(self.mfb)

        return

    def plot_decomposition(self):
        plt.plot(self.decomposed_data)
        plt.show()
        for i in range(self.decomposed_data.shape[1]):
            plt.subplot(8, 1, i + 1)
            plt.plot(self.decomposed_data[:, i])
        return plt.show()

    def filtered_constructor(self):
        for i in range(0, 4):
            np.append(self.reconstructed, self.decomposed_data[:, i])

        plt.plot(self.reconstructed)
        plt.show()
        return

    def iewt1d(self, ewt, mfb):
        real = all(np.isreal(ewt[:, 0]))
        if real:
            self.reconstructed = np.zeros(len(ewt[:, 0]))
            for i in range(0, 4):
                self.reconstructed += np.real(np.fft.ifft(np.fft.fft(ewt[:, i]) * mfb[::2, i]))
        else:
            self.reconstructed = np.zeros(len(ewt[:, 0])) * 0j
            for i in range(0, 4):
                self.reconstructed += np.fft.ifft(np.fft.fft(ewt[:, i]) * mfb[::2, i])
        return

    def plot_recon(self):
        plt.plot(self.reconstructed)
        plt.show()
