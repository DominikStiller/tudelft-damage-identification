from typing import Dict
import pandas as pd
import validclust as vld
import numpy as np


def find_optimal_number_of_clusters(features: pd.DataFrame, n_start, n_end) -> Dict[str, float]:
    """
    Find the optimal number of clusters k based on average of Davies-Bouldin, Silhouette and Dunn indexes.

    Args:
        features: the features of all examples
        n_start: start of the range of k's to try
        n_end: end of the range of k's to try
    Returns:
        Optimal number of clusters k
    """
    vclust = vld.ValidClust(k=list(range(n_start, n_end + 1)), methods=["kmeans", "hierarchical"])
    cvi_vals = vclust.fit_predict(features)
    indices = cvi_vals.to_numpy()

    # Compare all results and find the single optimal number
    maximize = [0, 1, 3, 4, 5, 7]
    minimize = [2, 6]

    # Normalize indices that need to be maximized
    for i in maximize:
        maximum = np.max(indices[i][:])
        minimum = np.min(indices[i][:])
        for j in range(n_end - n_start + 1):
            indices[i][j] = (indices[i][j] - minimum) / (maximum - minimum)

    # Normalize indices that need to be minimized
    for i in minimize:
        minimum = np.min(indices[i][:])
        maximum = np.max(indices[i][:])
        for j in range(n_end - n_start + 1):
            indices[i][j] = (indices[i][j] - maximum) / (minimum - maximum)

    # Find optimal k for each clustering method based on maximum normalized mean of indices
    kmeans_array = indices[:4]
    kmeansaverages = np.mean(kmeans_array, axis=0)
    kmeansindex = np.argmax(kmeansaverages)

    hierarchical_array = indices[4:]
    hierarchicalaverages = np.mean(hierarchical_array, axis=0)
    hierarchicalindex = np.argmax(hierarchicalaverages)

    overallaverages = np.mean(indices, axis=0)
    overallindex = np.argmax(overallaverages)

    return {
        "kmeans": kmeansindex + n_start,
        "hierarchical": hierarchicalindex + n_start,
        "fuzzy-cmeans": kmeansindex + n_start,
        "overall": overallindex + n_start,
    }
