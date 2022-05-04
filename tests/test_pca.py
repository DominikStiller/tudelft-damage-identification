from unittest import TestCase

import numpy as np

from damage_identification.pca import PrincipalComponents


class TestPrincipalComponents(TestCase):
    def test_get_n_components_default(self):
        params = {}
        self.assertEqual(
            PrincipalComponents._get_n_components(params),
            PrincipalComponents.DEFAULT_EXPLAINED_VARIANCE,
        )

    def test_get_n_components_explained_variance(self):
        params = {"explained_variance": 0.85}
        self.assertEqual(
            PrincipalComponents._get_n_components(params),
            0.85,
        )

    def test_get_n_components_n_components(self):
        params = {"n_principal_components": 3}
        self.assertEqual(
            PrincipalComponents._get_n_components(params),
            3,
        )

    def test_get_n_components_both(self):
        params = {"explained_variance": 0.85, "n_principal_components": 3}
        self.assertRaises(Exception, PrincipalComponents._get_n_components, params)

    def test_pca(self):
        example = np.array([[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        model = PrincipalComponents({"explained_variance": 0.9})
        model.train(example)
        result = model.transform(example)
        self.assertTrue(np.array_equal(result[-1], result[-2]))
