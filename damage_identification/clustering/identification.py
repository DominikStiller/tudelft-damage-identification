import pandas as pd

from damage_identification.damage_mode import DamageMode


def assign_damage_mode(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([predictions, features], axis=1)
    identifications = pd.DataFrame(index=data.index)

    for clusterer in predictions.columns:
        # Mapping from cluster index to damage mode
        mappings = {
            i: _identify_damage_mode(i, predictions[clusterer], features)
            for i in data[clusterer].unique()
        }
        assigned_damage_modes = list(filter(lambda e: e != DamageMode.UNKNOWN, mappings.values()))
        if len(set(assigned_damage_modes)) < len(assigned_damage_modes):
            print(f"WARNING: same damage mode was assigned to multiple clusters for {clusterer}")
        identifications[clusterer] = data[clusterer].map(mappings)

    return identifications


def _identify_damage_mode(
    cluster_index: int, predictions: pd.Series, features: pd.DataFrame
) -> DamageMode:
    """Identify the damage mode of a given cluster"""
    # TODO identify actual clusters based on cluster characteristics
    return DamageMode.UNKNOWN
