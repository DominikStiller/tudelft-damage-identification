from damage_identification.cli import parse_cli_args
from damage_identification.pipeline import Pipeline, PipelineMode


def main():
    print("### Damage mode identification tool ###")

    params = parse_cli_args()
    pipeline = Pipeline(params)

    if params["mode"] == PipelineMode.TRAINING:
        pipeline.run_training()
    elif params["mode"] == PipelineMode.PREDICTION:
        pipeline.run_prediction()
    elif params["mode"] == PipelineMode.EVALUATOIN:
        pipeline.run_evaluation()


if __name__ == "__main__":
    main()
