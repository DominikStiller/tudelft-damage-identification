import os
import pickle
from typing import Dict, Any

import numpy as np
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances_chunked
from damage_identification.evaluation.cluster_performace import gpu_dist_matrix
from fastdist import fastdist
from damage_identification.clustering.base import Clusterer
from numba import cuda
from numba import njit

class HierarchicalClusterer(Clusterer):
    """
    This class approximated hierarchical clustering through a KNN classifier.

    sklearn's hierachical clustering does not have a separate predict method and
    therefore always needs to retrain before prediction. Since this is inefficient,
    we do hierarchical clustering only during training of this class, and then approximate
    the cluster memberships through a KNN classifier, where each cluster corresponds
    to one classification label.

    Parameters:
        - n_clusters: number of clusters in HAC
        - n_neighbors: number of neighbors in KNN classifier
    """

    def __init__(self, params: Dict[str, Any]):
        """
        This method initializes the hierarchical clustering class and sets the amount of clusters that the data needs
        to be grouped into.
        For example: the n-dimensional data must be clustered into 3 distinct groups

        Args:
            params: parameters for the clustering method
        """
        self.model = None
        self.hcmodel = None
        super(HierarchicalClusterer, self).__init__("hierarchical", params)

    def save(self, directory):
        """
        Saves the hclust-knn state to an external file for extraction later on to predict which cluster a data point will be
        a part of.
        Note: empty state can also be saved externally.

        Args:
            directory: The location the dump file will be saved to.
        """
        with open(os.path.join(directory, "classifier.pickle"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(directory, "hclust.pickle"), "wb") as f:
            pickle.dump(self.hcmodel, f)

    def load(self, directory):
        """
        Loads the knn model object from an external file in the state it was saved. From this object the usual class
        methods can be used to extract information such as cluster center coordinates, etc.

        Args:
            directory: the directory where the dump file form the save function was saved.
        """
        with open(os.path.join(directory, "classifier.pickle"), "rb") as f:
            self.model = pickle.load(f)

    def train(self, data):
        """
        Clusters testdata and trains the KNN model.

        Args:
            data: data used to create the clusters to train the hclust model
        """
        if "n_neighbors" not in self.params:
            # Using an odd k = sqrt(n) is a best practice
            self.params["n_neighbors"] = round(np.sqrt(np.shape(data)[0]))
            if self.params["n_neighbors"] % 2 == 0:
                self.params["n_neighbors"] += 1
        labeled_data = self._do_hierarchical_clustering(data)
        self.model = KNeighborsClassifier(n_neighbors=self.params["n_neighbors"])
        self.model.fit(data, labeled_data)

    def _do_hierarchical_clustering(self, data):
        """Generate labels for data through hierarchical clustering"""
        '''if cuda.is_available():
            distmatrix = gpu_dist_matrix(data.to_numpy())
        else:'''
        print("computing slow distmatrix")
        shape = (np.shape(data)[0], np.shape(data)[0])
        #distmatrix = fastdist.matrix_to_matrix_distance(data.to_numpy(), data[0:3].to_numpy(), fastdist.euclidean, "euclidean")
        #distmatrix = split_distmatrix(data.to_numpy(), 1)
        with open('test.npy', 'ba+') as f:
            for chunk in pairwise_distances_chunked(data.to_numpy(), working_memory=4096, metric='euclidean'):
                chunk.tofile(f)

        distmatrix = np.memmap('test.npy', dtype=np.float64, mode='r+', shape=shape)
        print(distmatrix)
        print(np.shape(distmatrix))
        self.hcmodel = AgglomerativeClustering(n_clusters=self.params["n_clusters"], affinity='precomputed', linkage='single')
        labeled_data = self.hcmodel.fit_predict(distmatrix)
        return labeled_data

    def predict(self, data) -> int:
        """
        Uses the created KNN model to predict which cluster the data point is a part of.

        Args:
            data: datapoint for which the label should be predicted using the KNN classifier trained using the
            hierarchical clusters"""
        prediction = self.model.predict(data)[0]
        return prediction


'''@njit
def split_distmatrix(data, mem_size):
    if np.shape(data)[0]**2 > mem_size*150e6:
        n_chunks = int(mem_size*150e6/np.shape(data)[0])
    else:
        n_chunks = 1
    print(n_chunks)
    chunks = np.array_split(data, n_chunks)
    #distmatrix = np.array([[]])
    distmatrix = np.empty((np.shape(data)[0], np.shape(data)[0]))
    #for c in chunks:
    #    np.append(distmatrix, fastdist.matrix_to_matrix_distance(data, c, fastdist.euclidean, "euclidean").T)


    distmatrix = [fastdist.matrix_to_matrix_distance(data, c, fastdist.euclidean, "euclidean").tolist() for c in chunks]
    print(distmatrix[-1])
    print(np.shape(distmatrix[-1]))
    print(np.shape(distmatrix[0]))
    return distmatrix
'''