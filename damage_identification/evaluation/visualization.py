import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from damage_identification.evaluation.plot_helpers import save_plot, format_plot_3D, format_plot_2D


def visualize_clusters(data: pd.DataFrame, clusterer_names: list[str]):
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
        ax2.set_title(f"Features ({clusterer})",y=1.04)
        ax2.set_xlabel(features[0].replace("_", " "), labelpad=10)
        ax2.set_ylabel(features[1].replace("_", " "), labelpad=10)
        ax2.set_zlabel(features[2].replace("_", " "), labelpad=10)

        format_plot_3D()
        save_plot(f"clustering_visualization_{clusterer}", fig)


def visualize_cumulative_energy(data: pd.DataFrame, clusterer_names: list[str]):
    for clusterer in clusterer_names:
        array = np.array(data[clusterer]).astype("int")
        k = np.max(array) + 1
        arrayindex = []
        clusterenergy = []
        cumenergy = []
        energy = np.array(data["energy"])

        for i in range(k):
            buffer = np.array(np.where(array == i))
            arrayindex.append(buffer)
            clusterenergy.append(energy[arrayindex[i]])
            cumenergy.append(np.cumsum(clusterenergy[i]))
            plt.scatter(arrayindex[i], cumenergy[i], c="b")
            plt.xlabel("Index of waveform [-]")
            plt.ylabel("Cumulative energy [J]")
            plt.title(f"Cluster {i}")
            format_plot_2D()
            save_plot(f"energy_plot_{clusterer}_{i}", plt)
