import os.path
import pickle
import sys
from enum import auto, Enum
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from damage_identification.clustering.base import Clusterer
from damage_identification.clustering.kmeans import KmeansClusterer
from damage_identification.clustering.optimal_k import find_optimal_number_of_clusters
from damage_identification.damage_mode import DamageMode
from damage_identification.features.base import FeatureExtractor
from damage_identification.features.direct import DirectFeatureExtractor
from damage_identification.features.fourier import FourierExtractor
from damage_identification.io import load_uncompressed_data, load_compressed_data
from damage_identification.pca import PrincipalComponents
from damage_identification.visualization.clustering import ClusteringVisualization


class Pipeline:
    PIPELINE_PERSISTENCE_FOLDER = "data/pipeline/"

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.feature_extractors: List[FeatureExtractor] = [
            DirectFeatureExtractor(params),
            FourierExtractor(params),
        ]
        self.clusterers: List[Clusterer] = [KmeansClusterer(params)]
        self.pca = PrincipalComponents(params)
        self.visualization_clustering = ClusteringVisualization()

    def _load_data(self, param_name, limit=None) -> Tuple[np.ndarray, int]:
        filename: str = self.params[param_name]

        print("Loading data set...")

        if filename.endswith(".csv"):
            data = load_uncompressed_data(filename)
        elif filename.endswith(".tradb"):
            data = load_compressed_data(filename)
        else:
            raise Exception("Unsupported data file type")

        if limit is not None:
            data = data[:limit, :]

        n_examples = data.shape[0]

        return data, n_examples

    def _load_pipeline(self):
        with open(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, "params.pickle"), "rb") as f:
            self.params.update(pickle.load(f))

        for feature_extractor in self.feature_extractors:
            feature_extractor.load(
                os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, feature_extractor.name)
            )

        for clusterer in self.clusterers:
            clusterer.load(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, clusterer.name))

        self.pca.load(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, "pca"))

    def _save_params(self):
        os.makedirs(self.PIPELINE_PERSISTENCE_FOLDER, exist_ok=True)
        with open(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, "params.pickle"), "wb") as f:
            params_to_store = self.params.copy()
            del params_to_store["mode"]
            del params_to_store["training_data_file"]
            pickle.dump(params_to_store, f)

    def _save_component(self, name, component):
        save_directory = os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, name)
        os.makedirs(save_directory, exist_ok=True)
        component.save(save_directory)

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

    def _reduce_features(self, features: pd.DataFrame) -> pd.DataFrame:
        features_reduced = self.pca.transform(features)
        n_features_reduced = features_reduced.shape[1]
        features_reduced = pd.DataFrame(
            features_reduced, columns=[f"pca_{i + 1}" for i in range(n_features_reduced)]
        )
        return features_reduced

    def _predict(self, features, n_examples):
        print(f"Predicting clusters (k = {self.params['n_clusters']})...")
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

        # Train feature extractor
        print("Training feature extractors...")
        for feature_extractor in self.feature_extractors:
            feature_extractor.train(examples)
            self._save_component(feature_extractor.name, feature_extractor)
        print("-> Trained feature extractors")

        # Extract features for PCA training
        print("Extracting features for PCA training...")
        features = self._extract_features(examples, n_examples)
        features = features.dropna()

        # Normalize features
        # TODO run actual normalization
        features /= features.max()

        # Train PCA
        print("Training PCA...")
        self.pca.train(features)
        self._save_component("pca", self.pca)

        # Perform PCA for cluster training
        features_reduced = self._reduce_features(features)
        print("-> Trained PCA")

        # Find optimal number of clusters if desired by user
        if self.params["n_clusters"] == "auto":
            print("Finding optimal number of clusters...")
            self.params["n_clusters"] = find_optimal_number_of_clusters(
                features_reduced, self.params["n_clusters_start"], self.params["n_clusters_end"]
            )["kmeans"]
            # TODO possibly change kmeans to overall or make method-specific
            print(f"-> Found optimal number of clusters (k = {self.params['n_clusters']})")

        # Train clustering
        print("Training clusterers...")
        for clusterer in self.clusterers:
            clusterer.train(features_reduced)
            self._save_component(clusterer.name, clusterer)
        print("-> Trained clusterers")

        self._save_params()

        print("PIPELINE TRAINING COMPLETED")

    def run_prediction(self):
        print("Loading pipeline...")
        self._load_pipeline()
        print("-> Loaded trained pipeline")

        data, n_examples = self._load_data("prediction_data_file", 500)
        print(f"-> Loaded prediction data set ({n_examples} examples)")

        # TODO run filtering

        features = self._extract_features(data, n_examples)

        # TODO run actual normalization
        features /= features.max()

        features_reduced = self._reduce_features(features)

        predictions = self._predict(features_reduced, n_examples)

        self.visualization_clustering.visualize_kmeans(features_reduced, predictions)

        # TODO run cluster identification

    def run_evaluation(self):
        data, n_examples = self._load_data("evaluation_data_file")
        print(f"-> Loaded evaluation data set ({n_examples} examples)")

        # TODO add evaluation mode


class PipelineMode(Enum):
    TRAINING = auto()
    PREDICTION = auto()
    EVALUATION = auto()
