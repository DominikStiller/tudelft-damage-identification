from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import pickle
import os
import pandas as pd
import numpy as np
from validclust import dunn
from fastdist import fastdist


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
    #distmatrix = euclidean_distances(data)
    #distmatrix = fastdist.matrix_pairwise_distance(data.to_numpy(), fastdist.euclidean, "euclidean", return_matrix=True)
    #distmatrix = FCM._dist(data, data)
    dunnmetric = dunn(distmatrix, labels)
    calinski = calinski_harabasz_score(data, labels)
    return [davies, silhouette, dunnmetric, calinski]

def collate_metrics(data, directory , indices):
    """
    Args:
        data: the training data from PCA with reduced features
        dir: the folder of the saved pipeline e.g. "data/pipeline_default"

    Returns:
        DataFrame containing the performance indices for all the clusterers
    """
    distmatrix = fastdist.matrix_pairwise_distance(data.to_numpy(), fastdist.euclidean, "euclidean", return_matrix=True)
    print("calculated distance matrix!")
    k_labels = load_labels(os.path.join(directory, "kmeans/model.pickle"), indices)
    k_metrics = np.array(get_metrics(data, k_labels, distmatrix))
    f_labels = load_fcmeans_labels(os.path.join(directory, "fcmeans/fcmeans.pickle"), data)
    f_metrics = np.array(get_metrics(data, f_labels, distmatrix))
    h_labels = load_labels(os.path.join(directory, "hierarchical/hclust.pickle"), indices)
    h_metrics = np.array(get_metrics(data, h_labels, distmatrix))
    collated = np.vstack((k_metrics, f_metrics, h_metrics))

    return pd.DataFrame(collated, columns=['Davies', 'Silhouette', 'Dunn', 'Calinski-Harabasz'], index=['kmeans', 'fcmeans', "hierarchical"])
