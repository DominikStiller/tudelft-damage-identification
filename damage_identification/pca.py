import os
import pickle
from typing import Any

import pandas as pd
from sklearn.decomposition import PCA


class PrincipalComponents:
    DEFAULT_EXPLAINED_VARIANCE = 0.95

    def __init__(self, params: dict[str, Any]):
        """
        Initialize the PCA dimensionality reducer.

        Args:
            params containing {"explained_variance": float, "n_principal_components": int}
        """
        if params is None:
            params = {}

        self.pca = PCA(n_components=self._get_n_components(params), svd_solver="full")

    @classmethod
    def _get_n_components(cls, params):
        if ("explained_variance" in params) and ("n_principal_components" in params):
            raise Exception("You cannot both provide explained_variance and n_principal_components")
        if ("explained_variance" not in params) and ("n_principal_components" not in params):
            params["explained_variance"] = cls.DEFAULT_EXPLAINED_VARIANCE
        if "explained_variance" in params:
            return params["explained_variance"]
        else:
            return params["n_principal_components"]

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
    def feature_names(self):
        return self.pca.feature_names_in_

    @property
    def correlations(self):
        return self.pca.components_

    @property
    def n_components(self):
        return self.pca.n_components_

    @property
    def explained_variance(self):
        # Cumulative explained variance of selected principal components
        return self.pca.explained_variance_ratio_[: self.n_components].sum()
