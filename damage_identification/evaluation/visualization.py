import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator

from damage_identification.evaluation.plot_helpers import (
    save_plot,
    format_plot_3d,
    format_plot_2d,
)


def visualize_all(data: pd.DataFrame, clusterer_names: list[str], results_folder: str):
    visualize_clusters(data, clusterer_names, results_folder)
    visualize_cumulative_energy(data, clusterer_names, results_folder)
    visualize_force_displacement(data, results_folder)


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
            rasterized=True,
        )
        ax1.set_title(f"PCA ({_clusterer_display_name(clusterer)})", y=1.04)
        ax1.set_xlabel("PC 1 ", labelpad=10)
        ax1.set_ylabel("PC 2 ", labelpad=10)
        ax1.set_zlabel("PC 3 ", labelpad=10)

        features = ["duration [μs]", "peak_frequency [kHz]", "central_frequency [kHz]"]

        ax2.scatter3D(
            data[features[0]],
            data[features[1]],
            data[features[2]],
            c=data[clusterer].map(cmap),
            depthshade=False,
            rasterized=True,
        )
        ax2.set_title(f"Features ({_clusterer_display_name(clusterer)})", y=1.04)
        ax2.set_xlabel(features[0].replace("_", " "), labelpad=10)
        ax2.set_ylabel(features[1].replace("_", " "), labelpad=10)
        ax2.set_zlabel(features[2].replace("_", " "), labelpad=10)

        format_plot_3d()
        save_plot(results_folder, f"clustering_visualization_{clusterer}", fig)


def visualize_cumulative_energy(
    data: pd.DataFrame, clusterer_names: list[str], results_folder: str
):
    if "displacement" in data.columns:
        x_feature = "displacement"
        x_label = "Relative displacement [%]"
    else:
        x_feature = "index"
        x_label = "Index"
        print(
            "WARNING: no metadata provided, plotting cumulative energy vs index instead of displacement"
        )

    for clusterer in clusterer_names:
        predicted_clusters = data[clusterer].to_numpy()
        energy = data["energy"].to_numpy()
        displacement = data[x_feature].to_numpy()

        # Sort by displacement
        order_idx = displacement.argsort()
        energy = energy[order_idx]
        displacement = displacement[order_idx]

        for current_cluster in np.unique(predicted_clusters):
            idx_current_cluster = np.where(predicted_clusters == current_cluster)
            cumulative_energy = np.cumsum(energy[idx_current_cluster])

            # plt.figure(figsize=(475/320*8, 6))  # for DS-QI plots
            plt.scatter(
                displacement[idx_current_cluster] * 100, cumulative_energy, c="b", rasterized=True
            )

            plt.xlabel(x_label)
            plt.ylabel("Cumulative energy [aJ]")  # atto-Joule
            plt.yscale("log")
            plt.xlim(left=-5)

            format_plot_2d(ylocator=LogLocator(base=10, subs="all", numticks=100))
            save_plot(results_folder, f"energy_plot_{clusterer}_{current_cluster}")


def visualize_force_displacement(data: pd.DataFrame, results_folder: str):
    plt.figure(figsize=(6, 4))
    plt.plot(data["displacement"] * 100, data["force"] * 100, rasterized=True)

    plt.xlabel("Relative displacement [%]")
    plt.ylabel("Relative force [%]")
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    format_plot_2d()
    save_plot(results_folder, f"force_displacement")


def _clusterer_display_name(clusterer_name: str) -> str:
    return {
        "kmeans": "k-means",
        "fcmeans": "fuzzy c-means",
    }.get(clusterer_name, clusterer_name)


if __name__ == "__main__":
    import os
    import sys
    import pandas as pd

    results_folder = sys.argv[1]
    results_folder_new = os.path.join(
        os.path.dirname(results_folder), os.path.basename(results_folder) + "_new"
    )

    data = pd.read_pickle(os.path.join(results_folder, "data.pickle"))

    print("Generating plots...")
    clusterer_names = ["kmeans", "fcmeans", "hierarchical"]
    visualize_all(data, clusterer_names, results_folder_new)
    print(f"Saved plots to {results_folder_new}")
