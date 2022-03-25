import pandas as pd


class Normalization:
    """
    This class normalizes all the features from the direct feature extraction.

    The train function is used with a training dataset to determine maximum or minimum bounds to normalize to.
    These bounds can be saved with the save function, and loaded from a csv with the load function.
    The transform function can be used to normalize a new dataset based on the bounds of the training set.

    """

    def __init__(self):
        self.bounds = None

    def save(self, directory: str):
        """
        Saves the normalization bounds to a csv

        Args:
            directory: the directory to save the bounds to
        """

        self.bounds.to_csv(directory)

    def load(self, directory: str):
        """
        Loads the bounds of the normalization

        Args:
            directory: the directory to load the bounds from
        """

        self.bounds = pd.read_csv(directory, index_col=0)

    def train(self, train_data):
        """
        Creates bounds based on some training data

        Args:
            train_data: pandas dataframe of the training data

        Returns:
            bounds: pandas dataframe containing the min and max of each column of train_data
        """

        self.bounds = pd.DataFrame(columns=train_data.columns)
        self.bounds.loc["max"] = train_data.max()
        self.bounds.loc["min"] = train_data.min()

    def transform(self, data):
        """
        Transforms data based on the bounds from training data

        Args:
            data: data to be transformed

        Returns:
            normalize_data: pandas dataframe of the input data normalize between -1 and 1
        """

        normalize_data = (
            2 * (data - self.bounds.min()) / (self.bounds.max() - self.bounds.min()) - 1
        )
        return normalize_data
