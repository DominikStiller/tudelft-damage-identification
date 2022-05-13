import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from cluster_performace import collate_metrics
from sklearn.datasets import make_blobs
from validclust import ValidClust

def graph_metrics(directory):
    dirs = [os.path.join(directory, name) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    fig = plt.figure()
    minclusters = 2
    x = range(minclusters, len(dirs) + minclusters)
    k_metrics = np.zeros((len(dirs), 3))
    f_metrics = np.zeros((len(dirs), 3))
    h_metrics = np.zeros((len(dirs), 3))
    for i, d in enumerate(dirs):
        print(d)
        data = pd.read_pickle(os.path.join(d, "training_features_pca.pickle.bz2")).sample(n = 10000)
        indices = data.index
        metrics = collate_metrics(data, d, indices).to_numpy()
        k_metrics[i] = metrics[0]
        f_metrics[i] = metrics[1]
        h_metrics[i] = metrics[2]

    labels = ["Davies-Bouldin", "Silhouette", "Dunn"]
    for i in range(3):
       plt.scatter(x, k_metrics[:, i], label=labels[i])

    plt.xticks(x)
    plt.legend(loc=6)
    plt.ylabel("Index score")
    plt.xlabel("n clusters")
    plt.savefig("kmeans.png")
    plt.clf()

    for i in range(3):
        plt.scatter(x, f_metrics[:, i], label=labels[i])
    plt.xticks(x)
    plt.legend()
    plt.ylabel("Index score")
    plt.xlabel("n clusters")
    plt.savefig("fcmeans.png")
    plt.clf()

    for i in range(3):
        plt.scatter(x, h_metrics[:, i], label=labels[i])
    plt.xticks(x)
    plt.legend()
    plt.ylabel("Index score")
    plt.xlabel("n clusters")
    plt.savefig("hierarchical.png")


graph_metrics("data")
print("success!")



'''data = pd.read_pickle("data/pipeline_3clust/training_features_pca.pickle.bz2").sample(n = 20000)
print(data)

vclust = ValidClust(
    k=list(range(2, 4)),
    methods=['hierarchical', 'kmeans']
)
cvi_vals = vclust.fit_predict(data)
print(cvi_vals)
vclust.plot()
plt.show()'''