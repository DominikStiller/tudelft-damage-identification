import os.path
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
        self.clusterers: List[Clustering] = []

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
        all_features = []

        for example in data:
            features = {}
            for feature_extractor in self.feature_extractors:
                features.update(feature_extractor.extract_features(example))
            all_features.append(features)

        all_features = pd.DataFrame(all_features)

        return all_features

    def run_training(self):
        examples, n_examples = self._load_data("training_data_file")

        # TODO add filtering

        # Train feature extractor and save model
        for feature_extractor in self.feature_extractors:
            feature_extractor.train(examples)
            feature_extractor.save(
                os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, feature_extractor.name)
            )

        # Extract features to training of PCA and features
        examples_features = self._extract_features(examples)

        # TODO run PCA

        # Train clustering
        for clusterer in self.clusterers:
            clusterer.train(examples_features)
            clusterer.save(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, clusterer.name))

    def run_prediction(self):
        data, _ = self._load_data("prediction_data_file")

        # Load trained model
        for feature_extractor in self.feature_extractors:
            feature_extractor.load(
                os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, feature_extractor.name)
            )

        data_features = self._extract_features(data)

        # TODO run PCA

        predictions = []
        for clusterer in self.clusterers:
            clusterer.load(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, clusterer.name))
            # TODO change to iteration over single examples
            predictions.append(clusterer.predict(data_features))

    def run_evaluation(self):
        data, n_rows = self._load_data("evaluation_data_file")

        # TODO add evaluation mode


class PipelineMode(Enum):
    TRAINING = auto()
    PREDICTION = auto()
    EVALUATION = auto()
