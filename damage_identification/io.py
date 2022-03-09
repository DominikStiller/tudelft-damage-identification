import numpy as np
import os

def load_uncompressed_data(filename: str) -> np.ndarray:
    """
    Args:
        filename: data file name including file extension

    Returns:
        Numpy array of the data in the data file
    """
    data = np.loadtxt(f"data\{filename}", delimiter = ",")
    return data

def load_compressed_data(filename: str) -> np.ndarray:

    raise NotImplementedError()
