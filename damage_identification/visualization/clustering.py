import matplotlib.pyplot as plt
import pandas as pd


class ClusteringVisualization:
    """
    Performs visualization of clustered data based on a tuple of the first three primary components and their predicted
    damage modes

    Typical usage example:
        ClusteringVisualization().visualize_kmeans()
    """

    def __init__(self, data: tuple):
        self.pca, self.mode = data

    def classify_data(self, dmg_mode):
        """
        Takes the dataset, and extracts the datapoints that correspond to a certain damage mode.

        Args:
            dmg_mode: the requested damage mode to extract datapoints for.

        Returns:
            classed_data: pandas dataframe that contains the three PCA values, for every entry of the requested
            damage mode.
        """

        points = self.mode.loc[self.mode["mode_kmeans"] == dmg_mode]
        classed_data = pd.merge(self.pca, points, left_index=True, right_index=True)
        return classed_data

    def visualize_kmeans(self):
        """
        To visualize clustering from kmeans. Loops over all the damage modes present in the data, and plots the
        datapoints for each of these in a different
        colour, in 3D space.

        Args:
            self

        Returns:
            matplotlib visualization of the clusters
        """

        ax = plt.axes(projection="3d")
        clusters = self.mode["mode_kmeans"].drop_duplicates()
        colour = ["b", "g", "r", "c", "m"]
        colourpicker = 0
        for cluster in clusters:
            ax.scatter3D(
                self.classify_data(cluster)["pca_1"],
                self.classify_data(cluster)["pca_2"],
                self.classify_data(cluster)["pca_3"],
                c=colour[colourpicker],
            )
            colourpicker += 1
        ax.set_title("First three PCA directions - K-means")
        ax.set_xlabel("pca_1")
        ax.set_ylabel("pca_2")
        ax.set_zlabel("pca_3")
        plt.show()
