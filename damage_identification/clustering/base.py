from abc import ABC, abstractmethod

import pandas as pd

from damage_identification.damage_mode import DamageMode


class Clustering(ABC):
    """
    A base class for all clustering methods.
    """

    def __init__(self, params):
        """

        Args:
            params:
        """
        self.params = params

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the clustering method.

        This name will be used in log outputs and for state directories. Use a human-readable name without clustering
        as suffix. Examples are "kmeans", "hierarchical" or "som".
        """
        pass

    def save(self, directory):
        """
        Saves the state of the clustering method.

        This method should only save files to the directory specified in the argument.

        Args:
            directory: the directory to save the state to
        """
        pass

    def load(self, directory):
        """
        Loads the state of the clustering method.

        This method should only load files from the directory specified in the argument.

        Args:
            directory: the directory to load the state from
        """
        pass

    def train(self, examples: pd.DataFrame):
        """
        Train the clustering method.

        Args:
            examples: the training set with features of all training examples (shape n_examples x n_features)
        """
        pass

    def predict(self, example: pd.DataFrame) -> DamageMode:
        """
        Predict using the clustering method.

        Args:
            example: the features of a single example (shape 1 x n_features)

        Returns:
            The predicted damage mode
        """
        pass
