import matplotlib.pyplot as plt
import pandas as pd


class ClusteringVisualization:
    """
    Performs visualization of clustered data based on a tuple of the first three primary components and their predicted
    damage modes

    Typical usage example:
        ClusteringVisualization().visualize_kmeans()
    """

    def visualize(self, features: pd.DataFrame, modes: pd.DataFrame, clusterer: str):
        """
        To visualize clustering from kmeans. Loops over all the damage modes present in the data, and plots the
        datapoints for each of these in a different
        colour, in 3D space.

        Args:
            features: the features of each example after PCA (shape n_examples x n_features_reduced
            modes: the damage mode of each example predicted by each clusterer (shape n_examples x n_clusters)
            clusterer: the name of the clusterer to visualize
        """
        # Add dimensions if PCA components are not long enough
        for i in [1, 2, 3]:
            col = f"pca_{i}"
            if col not in features.columns:
                features[col] = 0

        ax = plt.axes(projection="3d")
        clusters = modes[clusterer].drop_duplicates()
        for cluster in clusters:
            current_features = self._classify_data(cluster, modes, features, clusterer)
            ax.scatter3D(
                current_features["pca_1"],
                current_features["pca_2"],
                current_features["pca_3"],
                depthshade=False,
            )
        ax.set_title(f"First three PCA directions - {clusterer}")
        ax.set_xlabel("pca_1")
        ax.set_ylabel("pca_2")
        ax.set_zlabel("pca_3")
        plt.show()

    def _classify_data(self, dmg_mode, modes, features, clusterer):
        """
        Takes the dataset, and extracts the datapoints that correspond to a certain damage mode.

        Args:
            dmg_mode: the requested damage mode to extract datapoints for.

        Returns:
            classed_data: pandas dataframe that contains the three PCA values, for every entry of the requested
            damage mode.
        """

        points = modes.loc[modes[clusterer] == dmg_mode]
        classed_data = pd.merge(features, points, left_index=True, right_index=True)
        return classed_data
