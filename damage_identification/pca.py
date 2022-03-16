from sklearn.decomposition import PCA
import pickle
import numpy as np
import os
from typing import Dict, Any


class PricipalComponents:
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the PCA dimensionality reductor.

        Args:
            name: name of the PCA reductor.
        """
        self.pca = PCA(n_components=params['explained_variance'])

    def save(self, directory):
        """
        Saves the state of the PCA.

        This method should only save files to the directory specified in the argument.

        Args:
            directory: the directory to save the state to
        """
        with open(os.path.join(directory, "pca.pickle"), "wb") as f:
            pickle.dump(self.pca, f)
        pass

    def load(self, directory):
        """
                Loads the state of the PCA.

                This method should only load files from the directory specified in the argument.

                Args:
                    directory: the directory to load the state from
        """
        with open(os.path.join(directory, "pca.pickle"), "rb") as f:
            self.pca = pickle.load(f)
        pass

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Reduce the feature dimension of an example based on principal components identified in train().

        Args:
            data: the single example (shape 1 x n_features)
        Returns:
            a NumPy array (shape 1 x n_features_reduced)
        """
        reduced = self.pca.transform(data)
        return reduced

    def train(self, data: np.ndarray):
        """
        Identify the principal components.

        Args:
            data: all data for training (shape n_examples x n_features)
            explained_variance: the desired explained variance to select the number of principal components to return (i.e. n_features_reduced)
        """
        self.pca = self.pca.fit(data)
        pass



principal = PricipalComponents({'explained_variance': 0.95})
X = np.random.rand(5000, 2000)
Y = np.random.rand(1, 2000)
principal.train(X)
print(np.shape(principal.transform(Y)))
