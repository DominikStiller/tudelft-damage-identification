from sklearn.decomposition import PCA
import numpy as np


class PricipalComponents():
    def __init__(self):
        """
        Initialize the PCA dimensionality reductor.

        Args:
            name: name of the PCA reductor.
        """
        pass

    def save(self, directory):
        """
        Saves the state of the PCA.

        This method should only save files to the directory specified in the argument.

        Args:
            directory: the directory to save the state to
        """
        pass

    def load(self, directory):
        """
        Loads the state of the PCA.

        This method should only load files from the directory specified in the argument.

        Args:
            directory: the directory to load the state from
        """
        pass

    def transform(self, data: np.ndarray):
        """
        Reduce the feature dimension of an example based on principal components identified in train().

        Args:
            data: the single example (shape 1 x n_features)
        """

        pass


    def train(self, data: np.ndarray, explained_variance: float) -> np.ndarray:
        """
        Identify the principal components.

        Args:
            data: all data for training (shape n_examples x n_features)
            explained_variance: the desired explained variance to select the number of principal components to return (i.e. n_features_reduced)
        Returns:
           a NumPy array (shape n_examples x n_features_reduced)
        """
        pca = PCA(n_components=explained_variance)
        pca.fit(data)
        reduced = pca.transform(data)
        return reduced

principal = PricipalComponents("PCA")
X = np.random.rand(5000, 2000)
print(np.shape(principal.train(X, 0.95)))
