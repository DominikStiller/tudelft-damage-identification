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

def collate_metrics(data):
    k_labels = load_labels("data/pipeline_comp0/kmeans/model.pickle")
    k_metrics = np.array(get_metrics(data, k_labels))
    #f_labels = load_data("data/pipeline_comp0/fcmeans/fcmeans.pickle")

    f_labels = load_fcmeans_labels("data/pipeline_comp0/fcmeans/fcmeans.pickle", data)
    f_metrics = np.array(get_metrics(data, f_labels))
    #h_labels = load_labels("data/pipeline_comp0/hclust/hclust.pickle")
    #h_metrics = np.array(get_metrics(data, h_labels)).T
    collated = np.vstack((k_metrics, f_metrics)) #h_metrics]
    return pd.DataFrame(collated)#, index=['Davies', 'Silhouette', 'Dunn'])#, columns=['kmeans', 'fcmeans', "hclust"])

testdata = pd.read_pickle("data/reduced_features_test.pickle").reset_index(drop=True)#.pop("index")
#print(testdata)
arr = collate_metrics(testdata)
print(arr)