import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from cluster_performace import collate_metrics, gpu_dist_matrix
from fastdist import fastdist
from numba import cuda


def graph_metrics(directory):
    dirs = [os.path.join(directory, name) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    fig = plt.figure()
    minclusters = 3
    x = range(minclusters, len(dirs) + minclusters)
    k_metrics = np.zeros((len(dirs), 4))
    f_metrics = np.zeros((len(dirs), 4))
    h_metrics = np.zeros((len(dirs), 4))
    data = pd.read_pickle(os.path.join("data/pipeline_3clust/training_features_pca.pickle.bz2"))
    if cuda.is_available():
        distmatrix = gpu_dist_matrix(data.to_numpy())
    else:
        distmatrix = fastdist.matrix_pairwise_distance(data.to_numpy(), fastdist.euclidean, "euclidean", return_matrix=True)
    print("calculated distance matrix!")
    for i, d in enumerate(dirs):
        print(d)
        indices = data.index
        metrics = collate_metrics(data, d, indices, distmatrix).to_numpy()
        print(metrics)
        k_metrics[i] = metrics[0]
        f_metrics[i] = metrics[1]
        h_metrics[i] = metrics[2]

    for i in [k_metrics, f_metrics, h_metrics]:
        i[:, 0] = np.abs((i[:, 0] - np.abs(np.max(i[:, 0]))) / np.max(i[:, 0]))
        i[:, 1] = (i[:, 1] + 1) * 0.5
        i[:, 3] = np.abs(
            (i[:, 3] - np.min(i[:, 3])) / (np.max(i[:, 3]) - np.min(i[:, 3])))

    labels = ["Davies-Bouldin", "Silhouette", "Dunn", "Calinski-Harabasz"]
    for i in range(len(labels)):
        plt.plot(x, k_metrics[:, i], label=labels[i], marker='o')

    plt.xticks(x)
    plt.legend(loc=6)
    plt.ylabel("Index score")
    plt.xlabel("n clusters")
    plt.savefig("kmeans.png")
    plt.clf()

    for i in range(len(labels)):
        plt.plot(x, f_metrics[:, i], label=labels[i], marker='o')
    plt.xticks(x)
    plt.legend()
    plt.ylabel("Index score")
    plt.xlabel("n clusters")
    plt.savefig("fcmeans.png")
    plt.clf()

    for i in range(len(labels)):
        plt.plot(x, h_metrics[:, i], label=labels[i], marker='o')
    plt.xticks(x)
    plt.legend()
    plt.ylabel("Index score")
    plt.xlabel("n clusters")
    plt.savefig("hierarchical.png")


graph_metrics("data")
print("success!")
