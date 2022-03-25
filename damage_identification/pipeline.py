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
from damage_identification.features.base import FeatureExtractor
from damage_identification.features.direct import DirectFeatureExtractor
from damage_identification.features.fourier import FourierExtractor
from damage_identification.features.normalization import Normalization
from damage_identification.io import load_uncompressed_data, load_compressed_data
from damage_identification.pca import PrincipalComponents
from damage_identification.preprocessing.wavelet_filtering import WaveletFiltering
from damage_identification.visualization.clustering import ClusteringVisualization


class Pipeline:
    PIPELINE_PERSISTENCE_FOLDER = "data/pipeline/"
    PER_RUN_PARAMS = ["mode", "training_data_file", "limit_data"]

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.feature_extractors: List[FeatureExtractor] = [
            DirectFeatureExtractor(params),
            FourierExtractor(params),
        ]
        self.clusterers: List[Clusterer] = [KmeansClusterer(params)]
        self.wavelet_filter = WaveletFiltering(params)
        self.normalization = Normalization()
        self.pca = PrincipalComponents(params)
        self.visualization_clustering = ClusteringVisualization()

    def _load_data(self, dataset_param_name) -> Tuple[np.ndarray, int]:
        """Load the dataset for the session"""
        filename: str = self.params[dataset_param_name]

        print("Loading data set...")

        if filename.endswith(".csv"):
            data = load_uncompressed_data(filename)
        elif filename.endswith(".tradb"):
            data = load_compressed_data(filename)
        else:
            raise Exception("Unsupported data file type")

        if "limit_data" in self.params:
            data = data[: self.params["limit_data"], :]

        n_examples = data.shape[0]

        return data, n_examples

    def _load_pipeline(self):
        """Load all components of a saved pipeline"""
        with open(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, "params.pickle"), "rb") as f:
            stored_params: Dict = pickle.load(f)
            self.params.update(stored_params)
            for k, v in stored_params.items():
                print(f" - {k}: {v}")

        for feature_extractor in self.feature_extractors:
            feature_extractor.load(
                os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, feature_extractor.name)
            )

        for clusterer in self.clusterers:
            clusterer.load(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, clusterer.name))

        self.normalization.load(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, "normalization"))
        self.pca.load(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, "pca"))

    def _save_pipeline(self):
        """Save all components of the pipeline"""

        def _save_dir(name):
            save_directory = os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, name)
            os.makedirs(save_directory, exist_ok=True)
            return save_directory

        for feature_extractor in self.feature_extractors:
            feature_extractor.save(_save_dir(feature_extractor.name))

        for clusterer in self.clusterers:
            clusterer.save(_save_dir(clusterer.name))

        self.normalization.save(_save_dir("normalization"))
        self.pca.save(_save_dir("pca"))

        # Save parameters
        with open(os.path.join(self.PIPELINE_PERSISTENCE_FOLDER, "params.pickle"), "wb") as f:
            params_to_store = self.params.copy()
            for param in self.PER_RUN_PARAMS:
                if param in params_to_store:
                    del params_to_store[param]
            pickle.dump(params_to_store, f)

    def _apply_wavelet_filtering(self, data: np.ndarray, n_examples) -> np.ndarray:
        if not self.params["skip_filter"]:
            print("Applying wavelet filtering...")
            with tqdm(total=n_examples, file=sys.stdout) as pbar:

                def do_filter(example):
                    pbar.update()
                    return self.wavelet_filter.filter_single(example)

                data = np.apply_along_axis(do_filter, axis=1, arr=data)
            print("-> Applied wavelet filtering")
        return data

    def _extract_features(self, data: np.ndarray, n_examples) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features using all feature extractors and combine into single DataFrame"""
        all_features = []
        n_invalid = 0
        valid_mask = pd.Series(index=np.arange(n_examples), dtype="boolean").fillna(True)

        print("Extracting features...")
        with tqdm(total=n_examples, file=sys.stdout) as pbar:
            for i, example in enumerate(data):
                features = {}
                for feature_extractor in self.feature_extractors:
                    features.update(feature_extractor.extract_features(example))

                # A None feature means that example is invalid
                if None in features.values():
                    valid_mask.iloc[i] = False
                    n_invalid += 1
                all_features.append(features)

                pbar.update()

        all_features = pd.DataFrame(all_features)
        print(f"-> Extracted features ({n_invalid} examples were invalid)")

        return all_features, valid_mask

    def _reduce_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA to all features"""
        features_reduced = self.pca.transform(features)
        n_features_reduced = features_reduced.shape[1]
        features_reduced = pd.DataFrame(
            features_reduced, columns=[f"pca_{i + 1}" for i in range(n_features_reduced)]
        )
        return features_reduced

    def _predict(self, features, n_examples):
        """Predict cluster memberships of all examples"""
        print(f"Predicting cluster memberships (k = {self.params['n_clusters']})...")
        with tqdm(total=n_examples, file=sys.stdout) as pbar:

            def do_predict(series):
                pbar.update()
                return clusterer.predict(series.to_frame().transpose())

            predictions = {clusterer.name: None for clusterer in self.clusterers}
            for clusterer in self.clusterers:
                predictions[clusterer.name] = features.apply(do_predict, axis=1)

        predictions = pd.concat(predictions, axis=1)
        print("-> Predicted cluster memberships")

        return predictions

    def run_training(self):
        print("Parameters:")
        for k, v in self.params.items():
            if k not in self.PER_RUN_PARAMS:
                print(f" - {k}: {v}")

        data, n_examples = self._load_data("training_data_file")
        print(f"-> Loaded training data set ({n_examples} examples)")

        # Apply wavelet filtering
        data = self._apply_wavelet_filtering(data, n_examples)

        # Train feature extractor
        print("Training feature extractors...")
        for feature_extractor in self.feature_extractors:
            feature_extractor.train(data)

        # Extract features for PCA training
        features, valid_mask = self._extract_features(data, n_examples)
        features_valid = features.loc[valid_mask]

        # Normalize features
        self.normalization.train(features_valid)
        features_normalized = self.normalization.transform(features_valid)
        print("-> Trained feature extractors")

        # Train PCA
        print("Training PCA...")
        self.pca.train(features_normalized)

        # Perform PCA for cluster training
        features_reduced = self._reduce_features(features_normalized)
        print(
            f"-> Trained PCA ({self.params['explained_variance']:.0%} of variance "
            f"explained by {self.pca.n_components} principal components)"
        )

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
        print("-> Trained clusterers")

        # Save at the end so all modifications of the params by components are stored
        #    (e.g. setting defaults or number of clusters)
        self._save_pipeline()

        print("PIPELINE TRAINING COMPLETED")

    def run_prediction(self):
        print("Loading pipeline...")
        self._load_pipeline()
        print("-> Loaded trained pipeline")

        data, n_examples = self._load_data("prediction_data_file")
        print(f"-> Loaded prediction data set ({n_examples} examples)")

        # Apply wavelet filtering
        data = self._apply_wavelet_filtering(data, n_examples)

        # Extract, normalize and reduce features
        features, valid_mask = self._extract_features(data, n_examples)
        features_valid = features.loc[valid_mask]
        features_normalized = self.normalization.transform(features_valid)
        features_reduced = self._reduce_features(features_normalized)

        # Make and visualize cluster predictions
        predictions = self._predict(features_reduced, n_examples)
        self.visualization_clustering.visualize_kmeans(features_reduced, predictions)

        # TODO run cluster identification and apply valid mask

    def run_evaluation(self):
        data, n_examples = self._load_data("evaluation_data_file")
        print(f"-> Loaded evaluation data set ({n_examples} examples)")

        # TODO add evaluation mode


class PipelineMode(Enum):
    TRAINING = auto()
    PREDICTION = auto()
    EVALUATION = auto()
