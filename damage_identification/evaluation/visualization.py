import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator

from damage_identification.evaluation.plot_helpers import save_plot, format_plot_3d, format_plot_2d


def visualize_clusters(data: pd.DataFrame, clusterer_names: list[str], results_folder: str):
    """import matplotlib
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
        ax1.set_title(f"PCA ({clusterer})", y=1.04)
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
        ax2.set_title(f"Features ({clusterer})", y=1.04)
        ax2.set_xlabel(features[0].replace("_", " "), labelpad=10)
        ax2.set_ylabel(features[1].replace("_", " "), labelpad=10)
        ax2.set_zlabel(features[2].replace("_", " "), labelpad=10)

        format_plot_3d()
        save_plot(results_folder, f"clustering_visualization_{clusterer}", fig)


def visualize_cumulative_energy(
    data: pd.DataFrame, clusterer_names: list[str], results_folder: str
):
    for clusterer in clusterer_names:
        predicted_clusters = data[clusterer].to_numpy()
        energy = data["energy"].to_numpy()
        displacement = data["displacement"].to_numpy()

        # Sort by displacement
        order_idx = displacement.argsort()
        energy = energy[order_idx]
        displacement = displacement[order_idx]

        for current_cluster in np.unique(predicted_clusters):
            idx_current_cluster = np.where(predicted_clusters == current_cluster)
            cumulative_energy = np.cumsum(energy[idx_current_cluster])
            plt.scatter(displacement[idx_current_cluster], cumulative_energy, c="b")

            plt.xlabel("Displacement [mm]")
            plt.ylabel("Cumulative energy [J]")
            plt.yscale("log")
            plt.title(f"Cluster {current_cluster}")

            format_plot_2d(ylocator=LogLocator(base=10, subs="all", numticks=100))
            save_plot(results_folder, f"energy_plot_{clusterer}_{current_cluster}", plt)
