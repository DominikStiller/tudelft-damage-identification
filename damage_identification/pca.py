import os
import pickle
from typing import Dict, Any

import pandas as pd
from sklearn.decomposition import PCA


class PrincipalComponents:
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the PCA dimensionality reducer.

        Args:
            params containing {'explained_variance' : float}
        """
        if params is None:
            params = {}
        if "explained_variance" not in params:
            params["explained_variance"] = 0.95
        if not 0 < params["explained_variance"] < 1:
            raise Exception("Explained variance has to be between 0 and 1")

        self.pca = PCA(n_components=params["explained_variance"], svd_solver="full")

    def save(self, directory):
        """
        Saves the state of the PCA.

        This method should only save files to the directory specified in the argument.

        Args:
            directory: the directory to save the state to
        """
        with open(os.path.join(directory, "pca.pickle"), "wb") as f:
            pickle.dump(self.pca, f)

    def load(self, directory):
        """
        Loads the state of the PCA.

        This method should only load files from the directory specified in the argument.

        Args:
            directory: the directory to load the state from
        """
        with open(os.path.join(directory, "pca.pickle"), "rb") as f:
            self.pca = pickle.load(f)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce the feature dimension of an example based on principal components identified in train().

        Args:
            data: the single example (shape 1 x n_features) or all examples (n_examples x n_features)
        Returns:
            a DataFrame (shape 1 x n_features_reduced) or all examples (n_examples x n_features)
        """
        return self.pca.transform(data)

    def train(self, data: pd.DataFrame):
        """
        Identify the principal components.

        Args:
            data: all data for training (shape n_examples x n_features)
        """
        self.pca = self.pca.fit(data)

    @property
    def n_components(self):
        return self.pca.n_components_
