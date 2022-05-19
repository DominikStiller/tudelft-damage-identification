from typing import Union

import numpy as np
import pandas as pd
import vallenae as vae


def load_data(filenames: Union[str, list[str]]) -> np.ndarray:
    """
    Load acoustic emission data from one or more files. The file type is detected through the file extension.

    Args:
        filenames: One or more filenames to load data from

    Returns:
        Numpy array of the data in the data files
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    data = []
    for filename in filenames:
        if filename.endswith(".csv"):
            data.append(load_data_from_csv(filename))
        elif filename.endswith(".tradb"):
            data.append(load_data_from_tradb(filename))
        elif filename.endswith(".npy"):
            data.append(load_data_from_numpy(filename))
        else:
            raise Exception("Unsupported data file type")

    return np.vstack(data)


def load_data_from_csv(filename: str) -> np.ndarray:
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


def load_data_from_tradb(filename: str) -> np.ndarray:
    """
    Args:
        filename: data file name including file extension

    Returns:
        Numpy array of the data in the data file: n_examples x n_samples
    """
    compressed_data = vae.io.TraDatabase(filename)
    data = np.vstack(compressed_data.read()["data"].to_numpy())
    return data


def load_data_from_numpy(filename: str) -> np.ndarray:
    """
    Args:
        filename: data file name including file extension

    Returns:
        Numpy array of the data in the data file: n_examples x n_samples
    """
    return np.load(filename)


def load_metadata(filenames: Union[str, list[str]]) -> pd.DataFrame:
    """
    Load metadata that have been generated with metadata.py

    Args:
        filenames: One or more filenames to load metadata from

    Returns:
        Pandas DataFrame of the metadata [n_examples x n_metadata]
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    metadata = []
    for filename in filenames:
        metadata.append(pd.read_pickle(filename))

    metadata = pd.concat(metadata)

    # Normalize displacement and force
    # The loaded values are only proportional to the real displacement/force
    metadata["displacement"] = -metadata["displacement"] / metadata["displacement"].abs().max()
    metadata["force"] = -metadata["force"] / metadata["force"].abs().max()

    return metadata
