import os

import numpy as np
import pandas as pd
import time


def load_uncompressed_data(filename: str) -> np.ndarray:
    """
    Args:
        filename: data file name including file extension

    Returns:
        Numpy array of the data in the data file
    """
    data = (pd.read_csv(os.path.join("data", filename))).to_numpy()
    return data


def load_compressed_data(filename: str) -> np.ndarray:
    """
    Will be updated when compression type is known

    Args:
        filename: data file name including file extension

    Returns:
        Numpy array of the data in the data file
    """
    data = (pd.read_csv(os.path.join("data", filename))).to_numpy()
    return data

print(load_compressed_data("Waveforms.zip"))
