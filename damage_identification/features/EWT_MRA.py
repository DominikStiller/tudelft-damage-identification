from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import ewtpy
from damage_identification import io

matplotlib.rcParams['figure.dpi'] = 300

# T = 1000
# t = np.arange(1,T+1)/T
# f = np.cos(2*np.pi*0.8*t) + 2*np.cos(2*np.pi*10*t)+0.8*np.cos(2*np.pi*100*t)
# ewt,  mfb ,boundaries = ewtpy.EWT1D(f, N = 3)
# plt.plot(f)
# plt.plot(ewt)
# plt.show()
class MultiResolutionAnalysis():
    """
    Description
    """

    # def __init__(self, params: Dict[str: Any]):
    def __init__(self):
        """
        Description
        """
        self.signal_data = []
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
        ewt, mfb, boundaries = ewtpy.EWT1D(self.signal_data[1, :])
        print(mfb, boundaries)
        plt.plot(ewt)
        plt.show()
        for i in range(ewt.shape[1]):
            plt.subplot(8, 1, i+1)
            plt.plot(ewt[:,i])

        return plt.show()
