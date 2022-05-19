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


def graph_metrics(pipeline_dirs: Union[str, list]):
    if isinstance(pipeline_dirs, str):
        pipeline_dirs = pipeline_dirs.split(",")

    n_pipelines = len(pipeline_dirs)

    n_clusters = []
    k_metrics = np.zeros((n_pipelines, 4))
    f_metrics = np.zeros((n_pipelines, 4))
    h_metrics = np.zeros((n_pipelines, 4))

    for i, d in enumerate(pipeline_dirs):
        print(f"Processing {d}...")
        with open(os.path.join(d, "params.pickle"), "rb") as f:
            n_clusters.append(pickle.load(f)["n_clusters"])

        data = pd.read_pickle(os.path.join(d, "training_features_pca.pickle.bz2")).sample(n=300)
        indices = data.index

        metrics = _collate_metrics(data, d, indices).to_numpy()

        k_metrics[i] = metrics[0]
        f_metrics[i] = metrics[1]
        h_metrics[i] = metrics[2]

    clusterer_names = ["kmeans", "fcmeans", "hierarchical"]
    for m, name in zip([k_metrics, f_metrics, h_metrics], clusterer_names):
        m[:, 0] = np.abs((m[:, 0] - np.abs(np.max(m[:, 0]))) / np.max(m[:, 0]))
        m[:, 1] = (m[:, 1] + 1) * 0.5
        m[:, 3] = np.abs((m[:, 3] - np.min(m[:, 3])) / (np.max(m[:, 3]) - np.min(m[:, 3])))

        fig = plt.figure(figsize=(12, 6))

        labels = ["Davies-Bouldin", "Silhouette", "Dunn", "Calinski-Harabasz"]
        for i in range(len(labels)):
            plt.plot(n_clusters, k_metrics[:, i], label=labels[i], marker="o")

        plt.xticks(n_clusters)
        plt.legend(loc=6)
        plt.ylabel("Index scores")
        plt.xlabel("k")

        format_plot_2d()
        save_plot("data/results/indexes", name, fig)
    print(f"Saved results to data/results/indexes")


def _collate_metrics(data, directory, indices):
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

    k_labels = _load_labels(os.path.join(directory, "kmeans/model.pickle"), indices)
    k_metrics = np.array(_get_metrics(data, k_labels, distmatrix))

    f_labels = _load_fcmeans_labels(os.path.join(directory, "fcmeans/fcmeans.pickle"), data)
    f_metrics = np.array(_get_metrics(data, f_labels, distmatrix))

    h_labels = _load_labels(os.path.join(directory, "hierarchical/hclust.pickle"), indices)
    h_metrics = np.array(_get_metrics(data, h_labels, distmatrix))

    return pd.DataFrame(
        np.vstack((k_metrics, f_metrics, h_metrics)),
        columns=["Davies", "Silhouette", "Dunn", "Calinski-Harabasz"],
        index=["kmeans", "fcmeans", "hierarchical"],
    )


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
    pipelines = sys.argv[1]
    graph_metrics(pipelines)
