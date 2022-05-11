from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import os
import pandas as pd
import numpy as np
from validclust import dunn


def load_labels(directory):
    with open(os.path.join(directory), "rb") as f:
        model = pickle.load(f)
    labeled_data = model.labels_
    return labeled_data

def load_fcmeans_labels(directory, data):
    with open(os.path.join(directory), "rb") as f:
        model = pickle.load(f)
    labeled_data = model.predict(data.to_numpy())
    return labeled_data

def get_metrics(data, labels):
    davies = davies_bouldin_score(data, labels)
    silhouette = silhouette_score(data, labels)
    distmatrix = euclidean_distances(data)
    dunnmetric = dunn(distmatrix, labels)
    return [davies, silhouette, dunnmetric]

def collate_metrics(data, dir):
    """
    Args:
        data: the training data from PCA with reduced features
        dir: the folder of the saved pipeline e.g. "data/pipeline_default"

    Returns:
        DataFrame containing the performance indeces for all the clsuterers
    """
    dir = "data/pipeline_test_performance"
    k_labels = load_labels(os.path.join(dir, "kmeans/model.pickle"))
    k_metrics = np.array(get_metrics(data, k_labels))
    f_labels = load_fcmeans_labels(os.path.join(dir, "fcmeans/fcmeans.pickle"), data)
    f_metrics = np.array(get_metrics(data, f_labels))
    h_labels = load_labels(os.path.join(dir, "hierarchical/hclust.pickle"))
    h_metrics = np.array(get_metrics(data, h_labels))
    collated = np.vstack((k_metrics, f_metrics, h_metrics))
    return pd.DataFrame(collated, columns=['Davies', 'Silhouette', 'Dunn'], index=['kmeans', 'fcmeans', "hierarchical"])

testdata = pd.read_pickle("data/test.pickle").reset_index(drop=True)
#print(testdata)
arr = collate_metrics(testdata)
print(arr)