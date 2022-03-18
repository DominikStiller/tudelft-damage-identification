from unittest import TestCase
import numpy as np
from damage_identification.pca import PrincipalComponents
import os

os.chdir("C:/Users/jakub/Desktop/Test")

class TestPCA(TestCase):
    def test_pca(self):
        example = np.array([[10, 1000, 1000, 0, 0], [0, 0, 10, 100, 1000], [10, 1000, 1000, 0, 0], [10, 1000, 1000, 0, 0]])
        model = PrincipalComponents({'explained_variance': 0.90})
        model.train(example)
        model.save("")
        model.load("")
        result = model.transform(example)
        print(len(result))
        print(result)
        # Expected value: 1000 Hz
        self.assertEqual(result[0], result[2])
