from abc import ABC, abstractmethod
from typing import Dict, Any

import pandas as pd

from damage_identification.damage_mode import DamageMode


class Clustering(ABC):
    """
    A base class for all clustering methods.
    """

    @abstractmethod
    def __init__(self, name: str, params: Dict[str, Any]):
        """
        Initialize the clustering method.

        The name will be used in log outputs and for state directories. Use a human-readable name without clustering
        as suffix. Examples are "kmeans", "hierarchical" or "som".

        Args:
            name: name of the clustering method
            params: parameters for the clustering method
        """
        self.name = name
        self.params = params

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

    @abstractmethod
    def predict(self, example: pd.DataFrame) -> DamageMode:
        """
        Predict using the clustering method.

        Args:
            example: the features of a single example (shape 1 x n_features)

        Returns:
            The predicted damage mode
        """
        pass
