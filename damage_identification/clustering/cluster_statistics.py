import pandas as pd


def print_cluster_statistics(
    predictions: pd.DataFrame, features_valid: pd.DataFrame, clusterer: str = "kmeans"
):
    assert predictions.shape[0] == features_valid.shape[0]
    data = pd.concat([predictions[clusterer], features_valid], axis=1).rename(
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

    data = data.groupby("cluster")

    print("\nCLUSTER STATISTICS")
    print("Means:")
    print(data.mean())
