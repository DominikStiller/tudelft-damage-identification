import numpy as np
import pandas as pd


def load_uncompressed_data(filename: str) -> np.ndarray:
    """
    Args:
        filename: data file name including file extension

    Returns:
        Numpy array of the data in the data file
    """
    data = np.transpose((pd.read_csv(filename, header=None)).to_numpy())
    # Remove rows that are only zeros
    data = data[~np.all(data == 0, axis=1)]
    return data


def load_compressed_data(filename: str) -> np.ndarray:
    """
    Args:
        filename: data file name including file extension

    Returns:
        Numpy array of the data in the data file: n_examples x n_samples
    """
    import vallenae as vae

    compressed_data = vae.io.TraDatabase(filename)
    data = np.vstack(compressed_data.read()["data"].to_numpy())
    return data
