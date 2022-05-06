import os
import pickle
from typing import Dict, Any, Optional

import pandas as pd
from fcmeans import FCM

from damage_identification.clustering.base import Clusterer


class FCMeansClusterer(Clusterer):
    def __init__(self, params: Dict[str, Any]):
        super(FCMeansClusterer, self).__init__("fcmeans", params)
        self.model: Optional[FCM] = None

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

    def train(self, examples: pd.DataFrame):
        self.model = FCM(n_clusters=self.params["n_clusters"], random_state=0)
        self.model.fit(examples.to_numpy())
        return self.model

    def predict(self, example: pd.DataFrame) -> int:
        prediction = self.model.predict(example.to_numpy())[0]
        return prediction
