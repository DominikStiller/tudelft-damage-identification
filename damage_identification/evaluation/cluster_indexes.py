import os
import pickle
import sys
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastdist import fastdist
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from validclust import dunn

from damage_identification.evaluation.plot_helpers import format_plot_2d, save_plot


METRICS_NAMES = ["Davies", "Silhouette", "Dunn", "Calinski-Harabasz"]


def graph_metrics(pipeline_dirs: Union[str, list]):
    if isinstance(pipeline_dirs, str):
        pipeline_dirs = pipeline_dirs.split(",")

    pipeline_names = []
    metrics = []

    for i, d in enumerate(pipeline_dirs):
        print(f"Processing {d}...")
        with open(os.path.join(d, "params.pickle"), "rb") as f:
            n_clusters = pickle.load(f)["n_clusters"]
            pipeline_names.append(os.path.basename(d).replace("pipeline_", ""))

        # Only use subset of data since distance matrix is too large otherwise
        # No sampling necessary since training data are shuffled
        data = pd.read_pickle(os.path.join(d, "training_features_pca.pickle.bz2")).head(n=30000)
        metrics.append(_collate_metrics(data, d, n_clusters))

    metrics = pd.concat(metrics)

    results_folder = os.path.join("data", "results", "indexes", "-".join(pipeline_names))

    for clusterer in metrics["clusterer"].unique():
        fig = plt.figure(figsize=(12, 6))

        for metric_name in METRICS_NAMES:
            m = metrics[
                (metrics["clusterer"] == clusterer) & (metrics["metric_name"] == metric_name)
            ]

            # Normalize indexes
            if metric_name == "Davies":
                m["metric_value"] = (
                    (m["metric_value"] - abs(m["metric_value"].max())) / m["metric_value"].max()
                ).abs()
            elif metric_name == "Silhouette":
                m["metric_value"] = (m["metric_value"] + 1) / 2
            elif metric_name == "Calinski-Harabasz":
                m["metric_value"] = (
                    (m["metric_value"] - m["metric_value"].min())
                    / (m["metric_value"].max() - m["metric_value"].min())
                ).abs()

            plt.plot(m["n_clusters"], m["metric_value"], label=metric_name, marker="o")

        plt.xticks(metrics["n_clusters"].unique())
        plt.legend(loc=6)
        plt.ylabel("Index scores")
        plt.xlabel("k")

        format_plot_2d()
        save_plot(results_folder, clusterer, fig)

    print(f"Saved results to {results_folder}")


def _collate_metrics(data, directory, n_clusters):
    """
    Args:
        data: the training data from PCA with reduced features
        dir: the folder of the saved pipeline e.g. "data/pipeline_default"

    Returns:
        DataFrame containing the performance indices for all the clusterers
    """
    distmatrix = fastdist.matrix_pairwise_distance(
        data.to_numpy(), fastdist.euclidean, "euclidean", return_matrix=True
    )

    metrics = []

    try:
        k_labels = _load_labels(os.path.join(directory, "kmeans/model.pickle"), data.index)
        metrics += [
            {"metric_name": METRICS_NAMES[i], "metric_value": val, "clusterer": "kmeans"}
            for i, val in enumerate(_get_metrics(data, k_labels, distmatrix))
        ]
    except:
        print("Failed to collect kmeans metrics")

    try:
        f_labels = _load_fcmeans_labels(os.path.join(directory, "fcmeans/fcmeans.pickle"), data)
        metrics += [
            {"metric_name": METRICS_NAMES[i], "metric_value": val, "clusterer": "fcmeans"}
            for i, val in enumerate(_get_metrics(data, f_labels, distmatrix))
        ]
    except:
        print("Failed to collect fcmeans metrics")

    try:
        h_labels = _load_labels(os.path.join(directory, "hierarchical/hclust.pickle"), data.index)
        metrics += [
            {"metric_name": METRICS_NAMES[i], "metric_value": val, "clusterer": "hierarchical"}
            for i, val in enumerate(_get_metrics(data, h_labels, distmatrix))
        ]
    except:
        print("Failed to collect hierarchical metrics")

    metrics = pd.DataFrame(metrics)
    metrics["n_clusters"] = n_clusters

    return metrics


def _load_labels(directory, indices):
    with open(os.path.join(directory), "rb") as f:
        model = pickle.load(f)
    labeled_data = model.labels_[indices]
    return labeled_data


def _load_fcmeans_labels(directory, data):
    with open(os.path.join(directory), "rb") as f:
        model = pickle.load(f)
    labeled_data = model.predict(data.to_numpy())
    return labeled_data


def _get_metrics(data, labels, distmatrix):
    davies = davies_bouldin_score(data, labels)
    silhouette = silhouette_score(data, labels)
    dunnmetric = dunn(distmatrix, labels)
    calinski = calinski_harabasz_score(data, labels)
    return [davies, silhouette, dunnmetric, calinski]


if __name__ == "__main__":
    # Prevent unnecessary warnings during index normalization
    pd.options.mode.chained_assignment = None

    pipelines = sys.argv[1]
    graph_metrics(pipelines)
