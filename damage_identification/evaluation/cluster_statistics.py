import pandas as pd


def print_cluster_statistics(data: pd.DataFrame):
    data = data.groupby("cluster")

    print("\nCLUSTER STATISTICS")
    print("COUNTS:")
    print(data.size().to_string())
    print("\nMEANS:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(data.mean())


def prepare_data_for_display(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    features_reduced: pd.DataFrame,
    clusterer: str = "kmeans",
) -> pd.DataFrame:
    data = pd.concat([predictions[clusterer], features, features_reduced], axis=1).rename(
        columns={clusterer: "cluster"}
    )

    # Change units for display
    data["duration"] *= 1e6
    data["rise_time"] *= 1e6
    data["peak_frequency"] /= 1e3
    data["central_frequency"] /= 1e3

    data = data.rename(
        columns={
            "duration": "duration [μs]",
            "rise_time": "rise_time [μs]",
            "peak_frequency": "peak_frequency [kHz]",
            "central_frequency": "central_frequency [kHz]",
        }
    )
    return data
