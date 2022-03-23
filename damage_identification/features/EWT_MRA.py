from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import ewtpy

from damage_identification.features.base import FeatureExtractor


# T = 1000
# t = np.arange(1,T+1)/T
# f = np.cos(2*np.pi*0.8*t) + 2*np.cos(2*np.pi*10*t)+0.8*np.cos(2*np.pi*100*t)
# ewt,  mfb ,boundaries = ewtpy.EWT1D(f, N = 3)
# plt.plot(f)
# plt.plot(ewt)
# plt.show()

class MultiResolutionAnalysis(FeatureExtractor):
    """
    Description
    """

    def __init__(self, params: Dict[str: Any]):
        """
        Description
        """
        super(MultiResolutionAnalysis, self).__init__("EWT_MRA", params)

    def load(self, directory):
        """
        Description
        """
        

    def EWT_MRA(self):
        """
        Description
        """