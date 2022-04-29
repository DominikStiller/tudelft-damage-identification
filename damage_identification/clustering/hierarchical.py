import numpy as np
import os
import pickle
from typing import Dict, Any
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import KNeighborsClassifier
from damage_identification.clustering.base import Clusterer

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

class HierarchicalClusterer(Clusterer):
    """
    This class Clusters the data according to the K-means clustering method

    Parameters:
        - n_clusters: number of clusters
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
        super(HierarchicalClusterer, self).__init__("hclust", params)

    def save(self, directory):
        """
        Saves the hclust state to an external file for extraction later on to predict which cluster a data point will be
        a part of.
        Note: empty state can also be saved externally.

        Args:
            directory: The location the dump file will be saved to.
        """
        with open(os.path.join(directory, "hclust.pickle"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, directory):
        """
        Loads the hclust model object from an external file in the state it was saved. From this object the usual class
        methods can be used to extract information such as cluster center coordinates, etc.

        Args:
            directory: the directory where the dump file form the save function was saved.
        """
        with open(os.path.join(directory, "hclust.pickle"), "rb") as f:
            self.model = pickle.load(f)

    def train(self, testdata):
        """
        Uses the testdata to train the hclust model. This creates the clusters and their centers.

        Args:
            testdata: data used to create the clusters to train the hclust model
        """
        #KMeansmodel = KMeans(self.params["n_clusters"], random_state=0)
        #KMeansmodel = KMeansmodel.fit(testdata)
        '''
        clusterer = AgglomerativeClustering(n_clusters=self.params["n_clusters"], linkage="ward")
        labeleddata = clusterer.fit_predict(testdata)
        self.model = KNeighborsClassifier(n_neighbors=self.params["n_neighbors"])
        self.model.fit(testdata, labeleddata)
        '''
        self.model = AgglomerativeClustering(n_clusters=self.params["n_clusters"], linkage="ward", compute_distances=True)
        self.model.fit(testdata)
        return self.model

    def predict(self, data) -> int:
        """
        Uses the created hclust model to predict which cluster the data point is a part of.

        Args:
            data: datapoint for which the label should be predicted using the created hierarchical clustering model
        """
        prediction = self.model.fit_predict(data)
        return prediction


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



data = pd.read_pickle('data\pca.pickle')
#print(data)
#clustering = AgglomerativeClustering().fit(data)
#print(clustering.fit_predict(data))


hc = HierarchicalClusterer({"n_clusters": 3, "n_neighbors": 5})
hct = hc.train(data)
#hcp = hc.predict(data)


plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(hct, truncate_mode="level", p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show().



