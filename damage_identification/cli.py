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

    # Fix skip_filter not being stored if absent even though store_true should handle this
    if "skip_filter" not in params:
        params["skip_filter"] = False

    return params


def print_cli_help():
    _construct_parser().print_help()


def _construct_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="python -m damage_identification",
        description="A tool for identifying the damage mode in CFRP composites using acoustic emissions. "
        "The tool has 3 modes: training, prediction and evaluation."
        "See https://github.com/DominikStiller/tudelft-damage-identification for more documentation.",
    )
    subparsers = parser.add_subparsers()

    # Parent parser for common parameters
    parser_params = ArgumentParser(add_help=False)
    parser_params.add_argument("--limit_data", type=int)
    parser_params.add_argument("data_file")

    # Training mode
    parser_training = subparsers.add_parser(
        "train", parents=[parser_params], argument_default=SUPPRESS
    )
    parser_training.set_defaults(mode=PipelineMode.TRAINING)

    parser_training.add_argument("--skip_filter", action="store_true")
    parser_training.add_argument("--skip_visualization", action="store_true")
    parser_training.add_argument("--wavelet_family", type=str)
    parser_training.add_argument("--wavelet_scale", type=int)
    parser_training.add_argument("--wavelet_threshold", type=str)
    parser_training.add_argument("--n_clusters", required=True)
    parser_training.add_argument("--direct_features_threshold", type=float)
    parser_training.add_argument("--direct_features_n_samples", type=int)
    parser_training.add_argument("--max_relative_peak_amplitude", type=float)
    parser_training.add_argument("--first_peak_domain", type=float)
    parser_training.add_argument("--explained_variance", type=float)

    # Prediction mode
    parser_prediction = subparsers.add_parser("predict", parents=[parser_params])
    parser_prediction.set_defaults(mode=PipelineMode.PREDICTION)

    # Evaluation mode
    parser_evaluation = subparsers.add_parser("evaluate", parents=[parser_params])
    parser_evaluation.set_defaults(mode=PipelineMode.EVALUATION)

    return parser
