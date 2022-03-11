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
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = KMeans(self.n_clusters, random_state=0)

    def save(self, directory):
        # print(directory)
        with open(os.path.join(directory, "model.pickle"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, directory):
        with open(os.path.join(directory, "model.pickle"), "rb") as f:
            self.model = pickle.load(f)

    def train(self, testdata):
        self.model = self.model.fit(testdata)
        return self.model

    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction


