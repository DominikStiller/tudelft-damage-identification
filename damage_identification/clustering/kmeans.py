import os
import pickle
from abc import ABC
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from damage_identification.clustering.base import Clustering


class KmeansClustering(Clustering):
    """
    This class Clusters the data according to the K-means clustering method
    
    """
    def __init__(self, name, n_clusters):
        self.n_clusters = n_clusters
        self.model = KMeans(self.n_clusters)
        self.name = name

    def save(self, directory):
        with open(os.path.join(directory, "model.pickle"), "wb") as f:
            pickle.dump(self.model, f)

    def train(self, testdata):
        kmeans = self.model
        kmeans.fit(testdata)
        print(kmeans.labels_, kmeans.cluster_centers_)
        return kmeans.labels_, kmeans.cluster_centers_

    def predict(self, data):
        kmeans = self.model
        print(kmeans.predict(data))
        return kmeans.predict(data)
