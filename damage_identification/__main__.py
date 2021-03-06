from damage_identification.cli import parse_cli_args, print_cli_help
from damage_identification.pipeline import Pipeline, PipelineMode


def main():
    params = parse_cli_args()
    pipeline = Pipeline(params)
    pipeline.initialize()

    if "mode" not in params:
        print_cli_help()
        return

    if params["mode"] == PipelineMode.TRAINING:
        pipeline.run_training()
    elif params["mode"] == PipelineMode.PREDICTION:
        pipeline.run_prediction()


if __name__ == "__main__":
    main()
