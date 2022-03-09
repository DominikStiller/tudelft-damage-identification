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


