from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class FeatureExtractor(ABC):
    """
    A base class for all feature extractors.
    """

    @abstractmethod
    def __init__(self, name: str, params: dict[str, Any]):
        """
        Initialize the feature extractor.

        The name will be used in log outputs and for state directories. Use a human-readable name without feature
        extractor as suffix. Examples are "fourier transform", "multiresolution analysis" or "autoencoder".

        Args:
            name: name of the feature extractor
            params: parameters for the feature extractor, uses default parameters if None
        """
        self.name = name
        self.params = params

    def save(self, directory):
        """
        Saves the state of a stateful feature extractor.

        Some feature extractors may require storing state, which means that they not process all examples independently
        but the result may actually depend on previous examples or training results. An example is the autoencoder,
        which is trained on a number of examples, and afterwards operates based on training results. During training,
        the optimal parameters are found, which are the state in this case.

        This method should only save files to the directory specified in the argument.

        This method does not have to be implemented for stateless feature extractors.

        Args:
            directory: the directory to save the state to
        """
        pass

    def load(self, directory):
        """
        Loads the state of a stateful feature extractor.

        This method should only load files from the directory specified in the argument.

        This method does not have to be implemented for stateless feature extractors.

        Args:
            directory: the directory to load the state from
        """
        pass

    def train(self, examples: np.ndarray):
        """
        Train the feature extractor if necessary.

        This method does not have to be implemented for stateless feature extractors.

        Args:
            examples: the training set with all training examples (shape n_examples_split x n_samples)
        """
        pass

    @abstractmethod
    def extract_features(self, example: np.ndarray) -> dict[str, Optional[float]]:
        """
        Extracts features from a single waveform.

        Args:
            example: a single example (shape 1 x n_samples)

        Returns:
            A dictionary containing items with each feature name value for the input example.
            Example: {"duration": 30.4, "average_amplitude": 3.7}
            An example can be marked as invalid by setting at least one of the features to None.
        """
        pass
