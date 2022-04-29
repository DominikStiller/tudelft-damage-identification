import matplotlib.pyplot as plt
import pandas as pd


class ClusteringVisualization:
    """
    Performs visualization of clustered data based on a tuple of the first three primary components and their predicted
    damage modes
    """

    def visualize(self, data: pd.DataFrame, clusterer_names: list[str]):
        """
        To visualize clustering from kmeans. Loops over all the damage modes present in the data, and plots the
        datapoints for each of these in a different
        colour, in 3D space.

        Args:
            data: the combined features and predictions
        """
        # Add dimensions if PCA components are not long enough
        for i in [1, 2, 3]:
            col = f"pca_{i}"
            if col not in data.columns:
                data[col] = 0

        cmap = plt.get_cmap("tab10")
        n_rows = len(clusterer_names)

        fig = plt.figure(figsize=(12, 6))

        for i, clusterer in enumerate(clusterer_names):
            ax1 = fig.add_subplot(n_rows, 2, 2 * i + 1, projection="3d")
            ax2 = fig.add_subplot(n_rows, 2, 2 * i + 2, projection="3d")

            ax1.scatter3D(
                data["pca_1"],
                data["pca_2"],
                data["pca_3"],
                c=data[clusterer].map(cmap),
                depthshade=False,
            )
            ax1.set_title(f"PCA ({clusterer})")
            ax1.set_xlabel("pca_1")
            ax1.set_ylabel("pca_2")
            ax1.set_zlabel("pca_3")

            # TODO check if contribute most to PCs
            features = ["duration [μs]", "peak_frequency [kHz]", "central_frequency [kHz]"]

            ax2.scatter3D(
                data[features[0]],
                data[features[1]],
                data[features[2]],
                c=data[clusterer].map(cmap),
                depthshade=False,
            )
            ax2.set_title(f"Features ({clusterer})")
            ax2.set_xlabel(features[0])
            ax2.set_ylabel(features[1])
            ax2.set_zlabel(features[2])

        plt.show()