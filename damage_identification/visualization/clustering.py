from random import choice
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from damage_identification.damage_mode import DamageMode


def generate_example_data():
    n_examples = 50
    n_pc = 3
    features = pd.DataFrame(
        np.random.rand(n_examples, n_pc), columns=[f"pca_{i + 1}" for i in range(0, n_pc)]
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


class ClusteringVisualization():
    """
    Performs visualization of clustered data based on a tuple of the first three primary components and their predicted damage modes

    Typical usage example:
        ClusteringVisualization().visualize_data()
    """
    def __init__(self, data: tuple):
        self.pca, self.mode = data

    def classify_data(self, dmg_mode):
        """
        Takes the dataset, and extracts the datapoints that correspond to a certain damage mode.

        Args:
            dmg_mode: the requested damage mode to extract datapoints for.

        Returns:
            a pandas dataframe that contains the three PCA values, for every entry of the requested damage mode.
        """

        points = self.mode.loc[self.mode["mode_kmeans"] == dmg_mode]
        classed_data = pd.merge(self.pca, points, left_index=True, right_index=True)
        return classed_data

    def visualize_kmeans(self):
        """
        To visualize clustering from kmeans. Loops over all the damage modes present in the data, and plots the datapoints for each of these in a different
        colour, in 3D space.

        Args: -

        Returns:-
        """

        ax = plt.axes(projection = "3d")
        dmgmodes = self.mode["mode_kmeans"]
        clusters = dmgmodes.drop_duplicates()
        colour = ["b", "g", "r", "c", "m"]
        colourpicker = 0
        for cluster in clusters:
            ax.scatter3D(self.classify_data(cluster)["pca_1"],
                         self.classify_data(cluster)["pca_2"],
                         self.classify_data(cluster)["pca_3"], c=colour[colourpicker])
            colourpicker += 1
        ax.set_title("First three PCA directions")
        ax.set_xlabel("pca_1")
        ax.set_ylabel("pca_2")
        ax.set_zlabel("pca_3")
        plt.show()


testdata = generate_example_data()
ClusteringVisualization(testdata).visualize_kmeans()

n_pc = 3
n_exampl
pcadata = pd.dataframe( ,columns=[f"pca_{i + 1}" for i in range(0, n_pc)])