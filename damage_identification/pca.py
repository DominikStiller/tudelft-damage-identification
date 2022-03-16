from sklearn.decomposition import PCA


class PricipalComponents():

    def __init__(self, name: str, params: Dict[str, Any]):
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
        pass