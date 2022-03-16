from random import choice
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from damage_identification.damage_mode import DamageMode


def generate_example_data():
    n_examples = 50
    n_pc = 3
    features = pd.DataFrame(
        np.random.rand(n_examples, n_pc), columns=[f"pca_{i+1}" for i in range(0, n_pc)]
    )
    predictions = pd.DataFrame(
        [
            choice(
                [
                    DamageMode.FIBER_PULLOUT,
                    DamageMode.MATRIX_CRACKING,
                    DamageMode.FIBER_MATRIX_DEBONDING,
                ]
            )
            for _ in range(0, n_examples)
        ],
        columns=["mode_kmeans"],
    )
    return features, predictions

testdata = generate_example_data()

def classify_data(dataset, dmg_mode):
    features, predictions = dataset
    points = predictions.loc[predictions["mode_kmeans"] == dmg_mode]
    classed_data = pd.merge(features, points, left_index=True, right_index=True)
    return classed_data

def visualize_data(data):
    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    dmgmodes = data[1]["mode_kmeans"]
    clusters = dmgmodes.drop_duplicates()
    colour = ["b", "g", "r", "c", "m"]
    colourpicker = 0
    for cluster in clusters:
        ax.scatter3D(classify_data(data, cluster)["pca_1"],
                     classify_data(data, cluster)["pca_2"],
                     classify_data(data, cluster)["pca_3"], c=colour[colourpicker])
        colourpicker += 1
    ax.set_title("First three PCA directions")
    ax.set_xlabel("pca_1")
    ax.set_ylabel("pca_2")
    ax.set_zlabel("pca_3")
    plt.show()

visualize_data(testdata)