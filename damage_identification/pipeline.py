import os.path
import sys
from enum import auto, Enum
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from damage_identification.clustering.base import Clustering
from damage_identification.clustering.kmeans import KmeansClustering
from damage_identification.damage_mode import DamageMode
from damage_identification.features.base import FeatureExtractor
from damage_identification.features.direct import DirectFeatureExtractor
from damage_identification.features.fourier import FourierExtractor
from damage_identification.io import load_uncompressed_data, load_compressed_data


class Pipeline:
    PIPELINE_PERSISTENCE_FOLDER = "data/pipeline/"

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.feature_extractors: List[FeatureExtractor] = [
            DirectFeatureExtractor(params),
            FourierExtractor(params),
        ]
        self.clusterers: List[Clustering] = [KmeansClustering(params)]

    def _load_data(self, param_name) -> Tuple[np.ndarray, int]:
        filename: str = self.params[param_name]

        print("Loading data set...")

        if filename.endswith(".csv"):
            data = load_uncompressed_data(filename)
        elif filename.endswith(".tradb"):
            data = load_compressed_data(filename)
        else:
            raise Exception("Unsupported data file type")

        n_examples = data.shape[0]

        return data, n_examples

    def _load_pipeline(self):
        for feature_extractor in self.feature_extractors:
            feature_extractor.load(
                os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, feature_extractor.name)
            )

        for clusterer in self.clusterers:
            clusterer.load(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, clusterer.name))

    def _extract_features(self, data: np.ndarray, n_examples) -> pd.DataFrame:
        all_features = []

        print("Extracting features...")
        with tqdm(total=n_examples, file=sys.stdout) as pbar:
            for i, example in enumerate(data):
                features = {}
                for feature_extractor in self.feature_extractors:
                    features.update(feature_extractor.extract_features(example))

                # A None feature means that example is invalid
                if None in features.values():
                    # Set all other features to None as well
                    features = dict.fromkeys(features, None)
                all_features.append(features)

                pbar.update()

        all_features = pd.DataFrame(all_features)
        print("-> Extracted features")

        return all_features

    def _predict(self, features, n_examples):
        print("Predicting clusters...")
        with tqdm(total=n_examples, file=sys.stdout) as pbar:

            def do_predict(series):
                pbar.update()
                # Skip prediction for invalid examples
                if series.isnull().any():
                    return DamageMode.INVALID
                else:
                    return clusterer.predict(series.to_frame().transpose())

            predictions = {clusterer.name: None for clusterer in self.clusterers}
            for clusterer in self.clusterers:
                predictions[clusterer.name] = features.apply(do_predict, axis=1)

        predictions = pd.concat(predictions, axis=1)
        print("-> Predicted clusters")

        return predictions

    def run_training(self):
        examples, n_examples = self._load_data("training_data_file")
        print(f"-> Loaded training data set ({n_examples} examples)")

        # TODO run filtering

        # Train feature extractor and save model
        print("Training feature extractors...")
        for feature_extractor in self.feature_extractors:
            feature_extractor.train(examples)

            save_directory = os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, feature_extractor.name)
            os.makedirs(save_directory, exist_ok=True)
            feature_extractor.save(save_directory)
        print("-> Trained feature extractors")

        # Extract features to training of PCA and features
        features = self._extract_features(examples, n_examples)
        features = features.dropna()

        # TODO run PCA training

        # Train clustering
        print("Training clusterers...")
        for clusterer in self.clusterers:
            clusterer.train(features)

            save_directory = os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, clusterer.name)
            os.makedirs(save_directory, exist_ok=True)
            clusterer.save(save_directory)
        print("-> Trained clusterers")

    def run_prediction(self):
        data, n_examples = self._load_data("prediction_data_file")
        print(f"-> Loaded prediction data set ({n_examples} examples)")

        self._load_pipeline()
        print("-> Loaded trained pipeline")

        # TODO run filtering

        features = self._extract_features(data, n_examples)

        # TODO run PCA

        predictions = self._predict(features, n_examples)

        print(predictions)

        # TODO run cluster identification

    def run_evaluation(self):
        data, n_examples = self._load_data("evaluation_data_file")
        print(f"-> Loaded evaluation data set ({n_examples} examples)")

        # TODO add evaluation mode


class PipelineMode(Enum):
    TRAINING = auto()
    PREDICTION = auto()
    EVALUATION = auto()
