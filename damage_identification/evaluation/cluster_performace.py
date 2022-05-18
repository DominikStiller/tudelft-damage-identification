from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import pickle
import os
import pandas as pd
import numpy as np
from validclust import dunn
from fastdist import fastdist
from numba import cuda

np_type = np.float64


def load_labels(directory, indices):
    with open(os.path.join(directory), "rb") as f:
        model = pickle.load(f)
    labeled_data = model.labels_[indices]
    return labeled_data


def load_fcmeans_labels(directory, data):
    with open(os.path.join(directory), "rb") as f:
        model = pickle.load(f)
    labeled_data = model.predict(data.to_numpy())
    return labeled_data


def get_metrics(data, labels, distmatrix):
    davies = davies_bouldin_score(data, labels)
    silhouette = silhouette_score(data, labels)
    dunnmetric = dunn(distmatrix, labels)
    calinski = calinski_harabasz_score(data, labels)
    return [davies, silhouette, dunnmetric, calinski]

def collate_metrics(data, directory , indices, distmatrix):
    """
    Args:
        data: the training data from PCA with reduced features
        dir: the folder of the saved pipeline e.g. "data/pipeline_default"

    Returns:
        DataFrame containing the performance indices for all the clusterers
    """

    k_labels = load_labels(os.path.join(directory, "kmeans/model.pickle"), indices)
    k_metrics = np.array(get_metrics(data, k_labels, distmatrix))
    f_labels = load_fcmeans_labels(os.path.join(directory, "fcmeans/fcmeans.pickle"), data)
    f_metrics = np.array(get_metrics(data, f_labels, distmatrix))
    h_labels = load_labels(os.path.join(directory, "hierarchical/hclust.pickle"), indices)
    h_metrics = np.array(get_metrics(data, h_labels, distmatrix))
    collated = np.vstack((k_metrics, f_metrics, h_metrics))
    return pd.DataFrame(collated, columns=['Davies', 'Silhouette', 'Dunn', 'Calinski-Harabasz'], index=['kmeans', 'fcmeans', "hierarchical"])


@cuda.jit("void(float{}[:, :], float{}[:, :])".format(64, 64))
def distance_matrix(mat, out):
    m = mat.shape[0]
    n = mat.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < m and j < m:
        for k in range(n):
            tmp = mat[i, k] - mat[j, k]
            d += tmp * tmp
        out[i, j] = d


def gpu_dist_matrix(mat):
    rows = mat.shape[0]

    block_dim = (16, 16)
    grid_dim = (int(rows/block_dim[0] + 1), int(rows/block_dim[1] + 1))

    stream = cuda.stream()
    mat2 = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
    out2 = cuda.device_array((rows, rows))
    distance_matrix[grid_dim, block_dim](mat2, out2)
    out = out2.copy_to_host(stream=stream)

    return out

