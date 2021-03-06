"""Parsing of command-line arguments"""
from argparse import ArgumentParser, SUPPRESS
from typing import Any

from damage_identification.pipeline import PipelineMode


def parse_cli_args() -> dict[str, Any]:
    params = vars(_construct_parser().parse_args())

    # Parse number of clusters (single number or find optimum in range
    if "n_clusters" in params:
        n_clusters = params["n_clusters"]
        if n_clusters.isdigit():
            params["n_clusters"] = int(n_clusters)
        elif "..." in n_clusters:
            params["n_clusters"] = "auto"
            params["n_clusters_start"] = int(n_clusters.split("...")[0])
            params["n_clusters_end"] = int(n_clusters.split("...")[1])
        else:
            raise "Invalid value for n_clusters"

    if "split" in params["data_file"]:
        print("WARNING: skipping filtering since operating on split dataset")
        params["skip_filter"] = True
        params["skip_saturation_detection"] = True

    if params["mode"] == PipelineMode.PREDICTION:
        print("WARNING: skipping shuffling by default since in prediction mode")
        params["skip_shuffling"] = True

    # Default params for argparse are not working for some reason
    # Therefore, set defaults manually here
    if "skip_filter" not in params:
        params["skip_filter"] = False
    if "skip_saturation_detection" not in params:
        params["skip_saturation_detection"] = False
    if "enable_peak_splitting" not in params:
        params["enable_peak_splitting"] = False
    if "skip_shuffling" not in params:
        params["skip_shuffling"] = False
    if "enable_identification" not in params:
        params["enable_identification"] = False
    if "sampling_rate" not in params:
        params["sampling_rate"] = 1000 * 2048  # 2048 samples per ms

    return params


def print_cli_help():
    _construct_parser().print_help()


def _construct_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="python -m damage_identification",
        description="A tool for identifying the damage mode in CFRP composites using acoustic emissions. "
        "The tool has 2 modes: training and prediction."
        "See https://github.com/DominikStiller/tudelft-damage-identification for more documentation.",
    )
    subparsers = parser.add_subparsers()

    # Parent parser for common parameters
    parser_params = ArgumentParser(add_help=False)
    parser_params.add_argument("--limit_data", type=int)
    parser_params.add_argument("--skip_shuffling", action="store_true")
    parser_params.add_argument("--enable_peak_splitting", action="store_true")
    parser_params.add_argument("-n", "--pipeline_name")
    parser_params.add_argument("data_file")

    # Training mode
    parser_training = subparsers.add_parser(
        "train", parents=[parser_params], argument_default=SUPPRESS
    )
    parser_training.set_defaults(mode=PipelineMode.TRAINING)

    parser_training.add_argument("--sampling_rate", type=float)
    parser_params.add_argument("--skip_filter", action="store_true")
    parser_params.add_argument("--skip_saturation_detection", action="store_true")
    parser_training.add_argument("--filtering_wavelet_family", type=str)
    parser_training.add_argument("--filtering_wavelet_scale", type=int)
    parser_training.add_argument("--filtering_wavelet_threshold", type=str)
    parser_training.add_argument("--mra_wavelet_family", type=str)
    parser_training.add_argument("--mra_wavelet_scale", type=int)
    parser_training.add_argument("--mra_time_bands", type=int)
    parser_training.add_argument("--mra_levels", type=int)
    parser_training.add_argument("--bandpass_low", type=float)
    parser_training.add_argument("--bandpass_high", type=float)
    parser_training.add_argument("--bandpass_order", type=int)
    parser_training.add_argument("--n_clusters", required=True)
    parser_training.add_argument("--direct_features_threshold", type=float)
    parser_training.add_argument("--direct_features_n_samples", type=int)
    parser_training.add_argument("--max_relative_peak_amplitude", type=float)
    parser_training.add_argument("--first_peak_domain", type=float)
    parser_training.add_argument("--explained_variance", type=float)
    parser_training.add_argument("--n_principal_components", type=int)
    parser_training.add_argument("--n_neighbours", type=int)

    # Prediction mode
    parser_prediction = subparsers.add_parser("predict", parents=[parser_params])
    parser_prediction.set_defaults(mode=PipelineMode.PREDICTION)

    parser_prediction.add_argument("--skip_visualization", action="store_true")
    parser_prediction.add_argument("--skip_statistics", action="store_true")
    parser_prediction.add_argument("--enable_identification", action="store_true")
    parser_prediction.add_argument("--metadata_file")

    return parser
