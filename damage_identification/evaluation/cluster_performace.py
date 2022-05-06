from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import os
import pandas as pd
import numpy as np
from validclust import dunn

def load_data(directory):
    with open(os.path.join(directory), "rb") as f:
        model = pickle.load(f)
    labeled_data = model.labels_
    return labeled_data

def get_metrics(data, labels):
    davies = davies_bouldin_score(data, labels)
    silhouette = silhouette_score(data, labels)
    distmatrix = euclidean_distances(data)
    dunnmetric = dunn(distmatrix, labels)
    return davies, silhouette, dunnmetric

def collate_metrics(clusterers, data):
    metrics = []
    k_labels = load_data("data/pipeline_default/kmeans")
    k_metrics = np.array(get_metrics(data, k_labels)).T
    f_labels = load_data("data/pipeline_default/fcmeans")
    f_metrics = np.array(get_metrics(data, f_labels)).T

    collated = [k]
    return pd.DataFrame(, columns=['kmeans', 'fcmeans', 'hierarchical'])

def make_dataframe():
    data = [["kmeans", ]]