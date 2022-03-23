"""Parsing of command-line arguments"""
from argparse import ArgumentParser, SUPPRESS
from typing import Any, Dict

from damage_identification.pipeline import PipelineMode


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

    # Training mode
    parser_training = subparsers.add_parser(
        "train", parents=[parser_params], argument_default=SUPPRESS
    )
    parser_training.set_defaults(mode=PipelineMode.TRAINING)
    parser_training.add_argument("training_data_file", metavar="data_file")

    parser_training.add_argument("--n_clusters", type=int, required=True)
    parser_training.add_argument("--direct_features_threshold", type=float)
    parser_training.add_argument("--direct_features_n_samples", type=int)
    parser_training.add_argument("--explained_variance", type=float)

    # Prediction mode
    parser_prediction = subparsers.add_parser("predict", parents=[parser_params])
    parser_prediction.set_defaults(mode=PipelineMode.PREDICTION)
    parser_prediction.add_argument("prediction_data_file", metavar="data_file")

    # Evaluation mode
    parser_evaluation = subparsers.add_parser("evaluate", parents=[parser_params])
    parser_evaluation.set_defaults(mode=PipelineMode.EVALUATION)
    parser_evaluation.add_argument("evaluation_data_file", metavar="data_file")

    return parser


def parse_cli_args() -> Dict[str, Any]:
    return vars(_construct_parser().parse_args())


def print_cli_help():
    _construct_parser().print_help()
