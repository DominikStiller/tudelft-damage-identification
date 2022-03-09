from enum import auto, Enum
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from damage_identification.clustering.base import Clustering
from damage_identification.features.base import FeatureExtractor
from damage_identification.features.direct import DirectFeatureExtractor
from damage_identification.io import load_uncompressed_data, load_compressed_data


class Pipeline:
    PIPELINE_PERSISTENCE_FOLDER = "data/pipeline/"

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.feature_extractors: List[FeatureExtractor] = [DirectFeatureExtractor(params)]
        self.clustering: List[Clustering] = []

    def _load_data(self, param_name) -> Tuple[np.ndarray, int]:
        filename: str = self.params[param_name]

        if filename.endswith(".csv"):
            data = load_uncompressed_data(filename)
        # TODO insert actual compressed file extension
        elif filename.endswith("."):
            data = load_compressed_data(filename)
        else:
            raise Exception("Unsupported data file type")

        n_rows = data.shape[0]

        return data, n_rows

    def _extract_features(self, data: np.ndarray) -> pd.DataFrame:
        all_features = None

        for example in data:
            features = {}
            for feature_extractor in self.feature_extractors:
                features.update(feature_extractor.extract_features(example))

            if all_features is None:
                all_features = pd.DataFrame(features, index=[0])
            else:
                all_features.append(features, ignore_index=True)

        all_features.reset_index()

        return all_features

    def run_training(self):
        examples, n_examples = self._load_data("training_data_file")

        # TODO add filtering

        # Train feature extractor and save model
        for feature_extractor in self.feature_extractors:
            feature_extractor.train(examples)
            feature_extractor.save(self.PIPELINE_PERSISTENCE_FOLDER)

        # Extract features to training of PCA and features
        all_features = self._extract_features(examples)

        print(all_features)

    def run_prediction(self):
        data, _ = self._load_data("prediction_data_file")

        # Load trained model
        for feature_extractor in self.feature_extractors:
            feature_extractor.load(self.PIPELINE_PERSISTENCE_FOLDER)

        all_features = self._extract_features(data)

    def run_evaluation(self):
        data, n_rows = self._load_data("evaluation_data_file")


class PipelineMode(Enum):
    TRAINING = auto()
    PREDICTION = auto()
    EVALUATOIN = auto()
