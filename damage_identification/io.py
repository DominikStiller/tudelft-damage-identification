import os
import numpy as np
import pandas as pd
import vallenae as vae

def load_uncompressed_data(filename: str) -> np.ndarray:
    """
    Args:
        filename: data file name including file extension

    Returns:
        Numpy array of the data in the data file
    """
    data = np.transpose((pd.read_csv(os.path.join("data", filename))).to_numpy())
    return data


def load_compressed_data(filename: str) -> np.ndarray:
    """
    Args:
        filename: data file name including file extension

    Returns:
        Numpy array of the data in the data file: n_examples x n_samples
    """
    compressed_data = vae.io.TraDatabase(os.path.join("data", filename))
    data = np.vstack(compressed_data.read()["data"].to_numpy())
    return data
