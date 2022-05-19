from typing import Optional
import os
import pandas as pd


def print_cluster_statistics(data: pd.DataFrame, clusterer_names: list[str], results_folder: str):

    file_name = "cluster_statistics.txt"
    cluster_statistics = open(os.path.join(results_folder, file_name), "w", encoding="utf-8")

    for clusterer in clusterer_names:
        cluster_statistics.write(f"\nCLUSTER STATISTICS ({clusterer})")
        data_grouped = data.rename(columns={clusterer: "cluster"}).groupby("cluster")

        cluster_statistics.write("COUNTS:")
        cluster_statistics.write(data_grouped.size().to_string())

        cluster_statistics.write("\nMEANS:")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            cluster_statistics.write(data_grouped.mean().to_string())

    cluster_statistics.close()


def prepare_data_for_display(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    features_reduced: pd.DataFrame,
    metadata: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepares the features and prediction results for display. Columns are renamed and units adjusted for
    human readability.

    Args:
        predictions: clustering predictions
        features: original features
        features_reduced: principal components of features
        metadata: metadata

    Returns:
        A tuple of the display dataframe and the list of clusterer names, which are columns in the display dataframe
    """
    data = [predictions, features, features_reduced]
    if metadata is not None:
        data.append(metadata)
    data = pd.concat(data, axis=1)

    # Add index and relative index as column
    data.reset_index(inplace=True)
    data.insert(1, "relative_index", data["index"] / len(data.index))

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

    clusterer_names = predictions.columns.tolist()

    return data, clusterer_names
