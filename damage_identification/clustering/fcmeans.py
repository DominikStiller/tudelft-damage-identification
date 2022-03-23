import os
import numpy as np
from fcmeans import FCM
from typing import Dict, Any
from damage_identification.clustering.base import Clusterer
import pickle


class FCMeansClusterer(Clusterer):
    def __init__(self, params: Dict[str, Any]):
        super(FCMeansClusterer, self).__init__("fcmeans", params)
        self.model = None

    def save(self, directory):
        """
        Saves the kmeans state to an external file for extraction later on to predict which cluster a data point will be
        a part of.
        Note: empty state can also be saved externally.

        Args:
            directory: The location the dump file will be saved to.
        """
        with open(os.path.join(directory, "fcmeans.pickle"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, directory):
        """
        Loads the kmeans model object from an external file in the state it was saved. From this object the usual class
        methods can be used to extract information such as cluster center coordinates, etc.

        Args:
            directory: the directory where the dump file form the save function was saved.
        """
        with open(os.path.join(directory, "fcmeans.pickle"), "rb") as f:
            self.model = pickle.load(f)

    def train(self, data: np.ndarray):
        self.model = FCM(n_clusters=self.params["n_clusters"], random_state=0)
        self.model.fit(data)

    def predict(self, data: np.ndarray) -> int:
        prediction = self.model.predict(data)
        return prediction
