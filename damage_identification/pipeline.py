"""
The top-level class connecting all components into a pipeline.

This class calls the methods of all components and forwards the data to the next component.
In between, some transformations (e.g. between NumPy and Pandas data structures) are necessary.
Loading and saving of the pipeline is also done here.

Data structures used through the pipeline:
- Raw waveforms (data): np.ndarray (shape n_examples x n_samples)
- Bandpass/wavelet-filtered waveforms (data_filtered): np.ndarray (shape n_examples x n_samples)
- Peak-split waveforms (data_split): np.ndarray (shape n_examples_split x n_samples)
- Raw features (features): pd.DataFrame (shape n_examples_split x n_features)
- Features of valid examples (features_valid): pd.DataFrame (shape n_examples_valid x n_features)
- Normalized features (features_normalized): pd.DataFrame (shape n_examples_valid x n_features)
- Reduces features after PCA (features_reduced): pd.DataFrame (shape n_examples_valid x n_features_reduced)
- Damage mode predictions (predictions): pd.DataFrame (shape n_examples_valid x n_clusterers)

The shapes are explained in the README.
"""
import os.path
import pickle
import sys
from enum import auto, Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from damage_identification.clustering.base import Clusterer
from damage_identification.clustering.fcmeans import FCMeansClusterer
from damage_identification.clustering.hierarchical import HierarchicalClusterer
from damage_identification.clustering.identification import assign_damage_mode
from damage_identification.clustering.kmeans import KmeansClusterer
from damage_identification.clustering.optimal_k import find_optimal_number_of_clusters
from damage_identification.damage_mode import DamageMode
from damage_identification.evaluation.statistics import (
    print_cluster_statistics,
    prepare_data_for_display,
)
from damage_identification.evaluation.visualization import (
    visualize_clusters,
    visualize_cumulative_energy,
)
from damage_identification.features.base import FeatureExtractor
from damage_identification.features.direct import DirectFeatureExtractor
from damage_identification.features.fourier import FourierExtractor
from damage_identification.features.mra import MultiResolutionAnalysisExtractor
from damage_identification.features.normalization import Normalization
from damage_identification.io import load_data, load_metadata
from damage_identification.pca import PrincipalComponents
from damage_identification.preprocessing.bandpass_filtering import BandpassFiltering
from damage_identification.preprocessing.peak_splitter import PeakSplitter
from damage_identification.preprocessing.saturation_detection import SaturationDetection
from damage_identification.preprocessing.wavelet_filtering import WaveletFiltering


class Pipeline:
    # Parameters that change with every execution and should not be saved
    PER_RUN_PARAMS = [
        "mode",
        "data_file",
        "limit_data",
        "pipeline_name",
        "skip_shuffling",
        "enable_peak_splitting",
    ]

    def __init__(self, params: dict[str, Any]):
        if ("pipeline_name" not in params) or (params["pipeline_name"] is None):
            params["pipeline_name"] = "default"
        self.params = params

        self.pipeline_persistence_folder = os.path.join(
            "data", f"pipeline_{self.params['pipeline_name']}"
        )

        self._initialize_components()

    def run_training(self):
        """Run the pipeline in training mode"""
        print("Parameters:")
        for k, v in self.params.items():
            print(f" - {k}: {v}")

        data, _, n_examples = self._load_data()

        # Apply bandpass and wavelet filtering
        data_filtered = self._apply_filtering(data, n_examples)

        # Split by peaks
        data_split, n_examples_split = self._split_by_peaks(data_filtered)

        # Train feature extractor
        print("Training feature extractors...")
        for feature_extractor in self.feature_extractors:
            feature_extractor.train(data_split)
        print("-> Trained feature extractors")

        # Extract features for PCA training
        features, valid_mask, _ = self._extract_features(data_split, n_examples_split)
        features_valid = features.loc[valid_mask]

        # Normalize features
        self.normalization.train(features_valid)
        features_normalized = self.normalization.transform(features_valid)

        # Train PCA
        print("Training PCA...")
        self.pca.train(features_normalized)

        # Perform PCA for cluster training
        features_reduced = self._reduce_features(features_normalized)
        print(
            f"-> Trained PCA ({self.pca.explained_variance:.0%} of variance "
            f"explained by {self.pca.n_components} principal components)"
        )

        # Save reduced features for cluster index analysis
        features_reduced.to_pickle(
            os.path.join(self.pipeline_persistence_folder, "training_features_pca.pickle.bz2"),
            compression="bz2",
        )

        # Find optimal number of clusters if desired by user
        if self.params["n_clusters"] == "auto":
            print("Finding optimal number of clusters...")
            self.params["n_clusters"] = find_optimal_number_of_clusters(
                features_reduced, self.params["n_clusters_start"], self.params["n_clusters_end"]
            )
            print(f"-> Found optimal number of clusters (k = {self.params['n_clusters']})")

        # Train clustering
        print("Training clusterers...")
        for clusterer in self.clusterers:
            clusterer.train(features_reduced)
        print("-> Trained clusterers")

        # Save at the end so all modifications of the params by components are saved
        #    (e.g. setting defaults or number of clusters)
        self._save_pipeline()

    def run_prediction(self):
        """Run the pipeline in prediction mode"""
        print("Loading pipeline...")
        self._load_pipeline()
        print("-> Loaded trained pipeline")

        data, metadata, n_examples = self._load_data()

        # Apply bandpass and wavelet filtering
        data_filtered = self._apply_filtering(data, n_examples)

        # Split by peaks
        data_split, n_examples_split = self._split_by_peaks(data_filtered)

        # Extract, normalize and reduce features
        features, valid_mask, n_valid_examples = self._extract_features(
            data_split, n_examples_split
        )
        features_valid = features.loc[valid_mask]
        metadata_valid = metadata.loc[valid_mask]
        features_normalized = self.normalization.transform(features_valid)
        features_reduced = self._reduce_features(features_normalized)

        # Make and visualize cluster predictions
        predictions = self._predict(features_reduced, n_valid_examples)

        data_display, clusterer_names = prepare_data_for_display(
            predictions, features_valid, features_reduced, metadata_valid
        )

        if not self.params["skip_statistics"]:
            print_cluster_statistics(data_display, clusterer_names)
            self.pca.print_correlations()
        if not self.params["skip_visualization"]:
            visualize_clusters(data_display, clusterer_names)
            visualize_cumulative_energy(data_display, clusterer_names)

        self._identify_damage_modes(predictions, features_valid, valid_mask)

    def _initialize_components(self):
        """Initialize all components including parameters"""
        # Pre-processing
        self.bandpass_filter = BandpassFiltering(self.params)
        self.wavelet_filter = WaveletFiltering(self.params)
        self.saturation_detection = SaturationDetection(self.params)

        # Feature extraction
        self.feature_extractors: list[FeatureExtractor] = [
            DirectFeatureExtractor(self.params),
            FourierExtractor(self.params),
            MultiResolutionAnalysisExtractor(self.params),
        ]
        self.normalization = Normalization()

        # PCA
        self.pca = PrincipalComponents(self.params)

        # Clustering
        self.clusterers: list[Clusterer] = [
            KmeansClusterer(self.params),
            FCMeansClusterer(self.params),
            HierarchicalClusterer(self.params),
        ]

    def _load_data(self) -> tuple[np.ndarray, Optional[pd.DataFrame], int]:
        """Load the dataset and optional metadata for the session"""
        filenames: list[str] = self.params["data_file"].split(",")

        if len(filenames) == 1:
            print("Loading dataset...")
        else:
            print(f"Loading {len(filenames)} datasets...")

        data = load_data(filenames)

        metadata = None
        if "metadata_file" in self.params and self.params["metadata_file"]:
            filenames: list[str] = self.params["metadata_file"].split(",")

            print("Loading metadata...")
            metadata = load_metadata(filenames)

            assert (
                metadata.shape[0] == data.shape[0]
            ), "Number of examples in data and metadata do not match"

        if not self.params["skip_shuffling"]:
            # Shuffle data and metadata in unison
            idx = np.arange(data.shape[0])
            np.random.shuffle(idx)

            data = data[idx]
            if metadata is not None:
                metadata = metadata.iloc[idx].reset_index(drop=True)

        if "limit_data" in self.params:
            data = data[: self.params["limit_data"], :]
            if metadata is not None:
                metadata = metadata.head(self.params["limit_data"])

        if not self.params["skip_saturation_detection"]:
            # Filter out saturated examples
            data_unsaturated, idx_unsaturated = self.saturation_detection.filter(data)
            if metadata is not None:
                metadata = metadata.iloc[idx_unsaturated].reset_index(drop=True)
        else:
            data_unsaturated = data

        n_examples = data_unsaturated.shape[0]
        n_saturated = data.shape[0] - n_examples

        print(f"-> Loaded dataset ({n_examples} examples, {n_saturated} were saturated)")

        return data_unsaturated, metadata, n_examples

    def _split_by_peaks(self, data: np.ndarray) -> tuple[np.ndarray, int]:
        # Split examples into two if multiple peaks are present
        if self.params["enable_peak_splitting"]:
            print("Splitting by peaks...")
            data_split, _, n_no_peaks, n_one_peak, n_over_two_peaks = PeakSplitter.split(data)

            print("Dataset contains")
            print(f" - {n_no_peaks} examples without peaks")
            print(f" - {n_one_peak} examples with one peak peaks")
            print(f" - {n_over_two_peaks} examples with two peaks or more")
            print("-> Split examples by peaks")
        else:
            data_split = data

        n_examples_split = data_split.shape[0]

        return data_split, n_examples_split

    def _load_pipeline(self):
        """Load all components of a saved pipeline"""
        with open(os.path.join(self.pipeline_persistence_folder, "params.pickle"), "rb") as f:
            saved_params: dict = pickle.load(f)
            self.params |= saved_params
            for k, v in self.params.items():
                print(f" - {k}: {v}")

        for feature_extractor in self.feature_extractors:
            feature_extractor.load(
                os.path.join(self.pipeline_persistence_folder, feature_extractor.name)
            )

        for clusterer in self.clusterers:
            clusterer.load(os.path.join(self.pipeline_persistence_folder, clusterer.name))

        self.normalization.load(os.path.join(self.pipeline_persistence_folder, "normalization"))
        self.pca.load(os.path.join(self.pipeline_persistence_folder, "pca"))

    def _save_pipeline(self):
        """Save all components of the pipeline"""
        for feature_extractor in self.feature_extractors:
            feature_extractor.save(self._create_component_dir(feature_extractor.name))

        for clusterer in self.clusterers:
            clusterer.save(self._create_component_dir(clusterer.name))

        self.normalization.save(self._create_component_dir("normalization"))
        self.pca.save(self._create_component_dir("pca"))

        # Save parameters
        with open(os.path.join(self.pipeline_persistence_folder, "params.pickle"), "wb") as f:
            params_to_save = self.params.copy()
            for param in self.PER_RUN_PARAMS:
                if param in params_to_save:
                    del params_to_save[param]
            pickle.dump(params_to_save, f)

    def _create_component_dir(self, name):
        save_directory = os.path.join(self.pipeline_persistence_folder, name)
        os.makedirs(save_directory, exist_ok=True)
        return save_directory

    def _apply_filtering(self, data: np.ndarray, n_examples) -> np.ndarray:
        """Apply bandpass and wavelet filtering to all examples"""
        if not self.params["skip_filter"]:
            print("Applying bandpass and wavelet filtering...")
            with tqdm(total=n_examples, file=sys.stdout) as pbar:

                def do_filter(example):
                    pbar.update()
                    example = self.bandpass_filter.filter_single(example)
                    example = self.wavelet_filter.filter_single(example)
                    return example

                data = np.apply_along_axis(do_filter, axis=1, arr=data)
            print("-> Applied bandpass and wavelet filtering")

        return data

    def _extract_features(
        self, data: np.ndarray, n_examples
    ) -> tuple[pd.DataFrame, pd.Series, int]:
        """Extract features using all feature extractors and combine into single DataFrame"""
        all_features = []
        n_invalid = 0
        valid_mask = pd.Series(index=np.arange(n_examples), dtype="boolean").fillna(True)

        print("Extracting features...")
        with tqdm(total=n_examples, file=sys.stdout) as pbar:
            for i, example in enumerate(data):
                features = {}
                for feature_extractor in self.feature_extractors:
                    features |= feature_extractor.extract_features(example)

                # A None feature means that example is invalid
                if None in features.values():
                    valid_mask.iloc[i] = False
                    n_invalid += 1
                all_features.append(features)

                pbar.update()

        all_features = pd.DataFrame(all_features)
        n_features = len(all_features.columns)
        n_valid = n_examples - n_invalid

        print(
            f"-> Extracted {n_features} features ({n_invalid} examples were invalid, {n_valid} were valid)"
        )

        return all_features, valid_mask, n_valid

    def _reduce_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA to all features"""
        features_reduced = self.pca.transform(features)

        n_features_reduced = features_reduced.shape[1]
        features_reduced = pd.DataFrame(
            features_reduced,
            columns=[f"pca_{i + 1}" for i in range(n_features_reduced)],
            index=features.index.copy(),
        )

        return features_reduced

    def _predict(self, features, n_examples) -> pd.DataFrame:
        """Predict cluster memberships of all examples"""
        print(f"Predicting cluster memberships (k = {self.params['n_clusters']})...")

        predictions = {clusterer.name: None for clusterer in self.clusterers}
        for clusterer in self.clusterers:
            with tqdm(total=n_examples, file=sys.stdout, desc=clusterer.name) as pbar:

                def do_predict(series):
                    pbar.update()
                    return clusterer.predict(series.to_frame().transpose())

                predictions[clusterer.name] = features.apply(do_predict, axis=1)

        predictions = pd.concat(predictions, axis=1).reindex(features.index.copy())
        print("-> Predicted cluster memberships")

        return predictions

    def _identify_damage_modes(
        self, predictions: pd.DataFrame, features: pd.DataFrame, valid_mask: pd.Series
    ):
        if not self.params["enable_identification"]:
            return

        identifications_valid = assign_damage_mode(predictions, features)

        identifications = pd.DataFrame(index=valid_mask.index)
        # Assign INVALID as damage mode to examples marked as invalid by a feature extractor
        identifications = pd.concat([identifications, identifications_valid], axis=1).fillna(
            DamageMode.INVALID
        )

        print("\nIDENTIFIED DAMAGE MODES")
        print(identifications)


class PipelineMode(Enum):
    TRAINING = auto()
    PREDICTION = auto()
