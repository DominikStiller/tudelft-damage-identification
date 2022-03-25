from unittest import TestCase

import numpy as np

from damage_identification.pca import PrincipalComponents


class TestPCA(TestCase):
    def test_pca(self):
        example = np.array([[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        model = PrincipalComponents({"explained_variance": 0.9})
        model.train(example)
        result = model.transform(example)
        self.assertTrue(np.array_equal(result[-1], result[-2]))
