import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from damage_identification.evaluation.save_plot import save_plot


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

        sb.set(
            context="paper",
            style="ticks",
            font_scale=1.6,
            font="sans-serif",
            rc={
                "lines.linewidth": 1.2,
                "axes.titleweight": "bold",
            },
        )

        # Add dimensions if PCA components are not long enough
        for i in [1, 2, 3]:
            col = f"pca_{i}"
            if col not in data.columns:
                data[col] = 0

        cmap = plt.get_cmap("tab10")

        for i, clusterer in enumerate(clusterer_names):
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")

            ax1.scatter3D(
                data["pca_1"],
                data["pca_2"],
                data["pca_3"],
                c=data[clusterer].map(cmap),
                depthshade=False,
            )
            ax1.set_title(f"PCA ({clusterer})")
            ax1.set_xlabel("pca 1", labelpad=10)
            ax1.set_ylabel("pca 2", labelpad=10)
            ax1.set_zlabel("pca 3", labelpad=10)

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
            ax2.set_xlabel(features[0].replace("_", " "), labelpad=10)
            ax2.set_ylabel(features[1].replace("_", " "), labelpad=10)
            ax2.set_zlabel(features[2].replace("_", " "), labelpad=10)

            fig.tight_layout(pad=2.5, h_pad=2, w_pad=0.2)
            save_plot(f"clustering_visualization_{clusterer}", fig)
