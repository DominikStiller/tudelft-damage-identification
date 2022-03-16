from random import choice

import numpy as np
import matplotlib as plt
import os

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
