"""Parsing of command-line arguments"""
from typing import Any, Dict

from damage_identification.pipeline import PipelineMode


def parse_cli_args() -> Dict[str, Any]:
    return {"mode": PipelineMode.TRAINING, "training_data_file": "Waveforms.csv"}
