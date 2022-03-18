import os
import numpy as np
from fcmeans import FCM
from typing import Dict, Any
from base import Clusterer
import pickle


class FuzzycmeansClusterer(Clusterer):
    def __init__(self, params: Dict[str, Any]):
        self.ncluster = params["n_clusters"]
        super(FuzzycmeansClusterer, self).__init__("cmeans", params)

    def gen_functions(self, data: np.ndarray):


    def predict(self, data: np.ndarray) -> int:

