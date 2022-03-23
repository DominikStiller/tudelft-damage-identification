from damage_identification.cli import parse_cli_args, print_cli_help
from damage_identification.pipeline import Pipeline, PipelineMode


def main():
    print("### Damage mode identification tool ###")

    params = parse_cli_args()
    pipeline = Pipeline(params)

    if "mode" not in params:
        print_cli_help()
        return

    if params["mode"] == PipelineMode.TRAINING:
        pipeline.run_training()
    elif params["mode"] == PipelineMode.PREDICTION:
        pipeline.run_prediction()
    elif params["mode"] == PipelineMode.EVALUATION:
        pipeline.run_evaluation()


if __name__ == "__main__":
    main()
