"""Parsing of command-line arguments"""
import argparse
from typing import Any, Dict

from damage_identification.pipeline import PipelineMode


def parse_cli_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="python -m damage_identification", description="Process some integers."
    )
    subparsers = parser.add_subparsers()

    # Training mode
    parser_training = subparsers.add_parser("train")
    parser_training.set_defaults(mode=PipelineMode.TRAINING)
    parser_training.add_argument("training_data_file", metavar="data_file")

    # Prediction mode
    parser_prediction = subparsers.add_parser("predict")
    parser_prediction.set_defaults(mode=PipelineMode.PREDICTION)
    parser_prediction.add_argument("prediction_data_file", metavar="data_file")

    # Evaluation mode
    parser_evaluation = subparsers.add_parser("evaluate")
    parser_evaluation.set_defaults(mode=PipelineMode.EVALUATION)
    parser_evaluation.add_argument("evaluation_data_file", metavar="data_file")

    params = vars(parser.parse_args())
    return params
