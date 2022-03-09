import numpy as np
import pandas as pd

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
    """
        Args:
            filename: data file name including file extension

        Returns:
            Numpy array of the data in the data file
        """
    data = (pd.read_csv(f"data\{filename}", compression = "zip")).to_numpy()
    return data
