from unittest import TestCase
import pandas as pd
from pandas._testing import assert_frame_equal

from damage_identification.features.normalization import Normalization


class TestNormalization(TestCase):
    def test_normalize_features(self):
        train_data = pd.DataFrame({"test": [1, 11]})

        test_data = pd.DataFrame({"test": [8]})

        confirmation_data = pd.DataFrame({"test": [0.4]})

        normalized_features = Normalization()
        normalized_features.train(train_data)
        transformed_data = normalized_features.transform(test_data)
        assert_frame_equal(transformed_data, confirmation_data)
