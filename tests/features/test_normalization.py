import tempfile
from unittest import TestCase

import pandas as pd
from pandas._testing import assert_frame_equal

from damage_identification.features.normalization import Normalization


class TestNormalization(TestCase):
    def test_normalize_features(self):
        train_data = pd.DataFrame({"test1": [1, 11], "test2": [0, 2]})

        test_data = pd.DataFrame({"test1": [8], "test2": [1]})

        expected = pd.DataFrame({"test1": [0.4], "test2": [0]})

        normalized_features = Normalization()
        normalized_features.train(train_data)
        actual = normalized_features.transform(test_data)

        assert_frame_equal(actual, expected, check_dtype=False)

    def test_load_save(self):
        train_data = pd.DataFrame({"test1": [1, 11], "test2": [0, 2]})

        presave = Normalization()
        presave.train(train_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            presave.save(tmpdir)
            postsave = Normalization()
            postsave.load(tmpdir)

        assert_frame_equal(presave.bounds, postsave.bounds)
