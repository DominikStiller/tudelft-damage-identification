from unittest import TestCase
import numpy as np
from damage_identification.pca import PrincipalComponents
import os

os.chdir("C:/Users/jakub/Desktop/Test")


class TestPCA(TestCase):
    def test_pca(self):
        example = np.array([[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        model = PrincipalComponents({"explained_variance": 0.9})
        model.train(example)
        model.save("")
        model.load("")
        result = model.transform(example)
        # print(len(result[0]))
        # Expected value: 1000 Hz
        self.assertTrue(np.array_equal(result[-1], result[-2]))
