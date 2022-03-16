
import pandas as pd

def find_optimal_number_of_clusters(features: pd.DataFrame, n_start, n_end) -> int:
    """
    Find the optimal number of clusters based on a voting scheme of Davies-Bouldin, Silhouette and Dunn indexes.

    Args:
        features: the features of all examples
        n_start: start of the range of k's to try
        n_end: end of the range of k's to try
    Returns:
        Optimal number of clusters
    """
    # Call validclust
    # Compare all results and find the single optimal number
    pass