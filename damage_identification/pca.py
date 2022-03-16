from sklearn.decomposition import PCA
import numpy as np


class PricipalComponents():
    def __init__(self, name: str):
        """
        Initialize the PCA dimensionality reductor.

        Args:
            name: name of the PCA reductor.
            params: parameters for the reductor
        """
        self.name = name

    def transform(data: np.ndarray) -> np.ndarray:
        """
        Reduce the feature dimension of an example based on principal components identified in train().

        Args:
            data: the single example (shape 1 x n_features)
        Returns:
            a NumPy array (shape 1 x n_features_reduced)
        """


        pass


    def train(data: np.ndarray, explained_variance: float) -> np.ndarray:


        """
        Identify the principal components.

        Args:
            data: all data for training (shape n_examples x n_features)
            explained_variance: the desired explained variance to select the number of principal components to return (i.e. n_features_reduced)
        Returns:
           a NumPy array (shape n_examples x n_features_reduced)
        """
        pca = PCA(n_components=2)
        print(pca.explained_variance_)

        pass

principal = PricipalComponents("PCA")
principal.train(np.array([1,0,1,0,1]), 0)
