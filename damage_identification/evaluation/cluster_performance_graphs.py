import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from cluster_performace import collate_metrics

def graph_metrics(directory):
    dirs = [os.path.join(directory, name) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    fig = plt.figure()
    minclusters = 3
    x = range(minclusters, len(dirs) + minclusters)
    k_metrics = np.zeros((len(dirs), 3))
    f_metrics = np.zeros((len(dirs), 3))
    h_metrics = np.zeros((len(dirs), 3))
    for i, d in enumerate(dirs):
        data = pd.read_pickle(os.path.join(d, "training_features_pca.pickle.bz2"))
        k_metrics[i] = collate_metrics(data, d).to_numpy()[0]
        f_metrics[i] = collate_metrics(data, d).to_numpy()[1]
        h_metrics[i] = collate_metrics(data, d).to_numpy()[2]

    labels = ["Davies-Bouldin", "Silhouette", "Dunn"]
    for i in range(3):
       plt.scatter(x, k_metrics[:, i], label=labels[i])

    plt.xticks(x)
    plt.legend()
    plt.savefig("kmeans.png")
    plt.clf()


    for i in range(3):
        plt.scatter(x, f_metrics[:, i], label=labels[i])
    plt.xticks(x)
    plt.legend()
    plt.savefig("fcmeans.png")
    plt.clf()

    for i in range(3):
        plt.scatter(x, h_metrics[:, i], label=labels[i])
    plt.xticks(x)
    plt.legend()
    plt.savefig("hierarchical.png")


graph_metrics("data")