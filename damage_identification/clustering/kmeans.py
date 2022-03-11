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
        """
        This method initializes the kmeans class and sets the amount of clusters that the data needs to be grouped into.
        For example: the n-dimensional data must be clustered into 3 distinct groups

        Args:
            n_clusters: number of clusters
        """
        self.n_clusters = n_clusters
        self.model = KMeans(self.n_clusters, random_state=0)

    def save(self, directory):
        """
        Saves the kmeans state to an external file for extraction later on to predict which cluster a data point will be
        a part of.
        Note: empty state can also be saved externally.

        Args:
            directory: The location the dump file will be saved to.
        """
        with open(os.path.join(directory, "model.pickle"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, directory):
        """
        Loads the kmeans model object from an external file in the state it was saved. From this object the usual class
        methods can be used to extract information such as cluster center coordinates, etc.

        Args:
            directory: the directory where the dump file form the save function was saved.
        """
        with open(os.path.join(directory, "model.pickle"), "rb") as f:
            self.model = pickle.load(f)

    def train(self, testdata):
        """
        Uses the testdata to train the kmeans model. This creates the clusters and their centers.

        Args:
            testdata: data used to create the clusters to train the kmeans model
        """
        self.model = self.model.fit(testdata)
        return self.model

    def predict(self, data):
        """
        Uses the created kmeans model to predict which cluster the data point is a part of.

        Args:
            data: datapoint(S) for which the label should be predicted using the created kmeans model
        """
        prediction = self.model.predict(data)
        return prediction
