"""This scripts extracts metadata such as time and displacement for AE datasets"""
import os
import sys

import numpy as np
import vallenae as vae
from scipy.interpolate import interp1d

from damage_identification.io import load_data

if __name__ == "__main__":
    filename_data = sys.argv[1]
    filename_metadata = sys.argv[2]

    filename_out = os.path.splitext(filename_data)[0].replace("_idx", "") + "_metadata.pickle"

    # Load index, either from index file (after splitting) or from original data file
    if filename_data.endswith("_idx.npy"):
        idx = np.load(filename_data)
    else:
        idx = np.arange(load_data(filename_data).shape[0])

    # Load corresponding .pridb file containing metadata
    if filename_metadata.endswith(".pridb"):
        pridb = vae.io.PriDatabase(filename_metadata)
    else:
        raise Exception("Second argument needs to be a .pridb file")

    # Parameter mapping depends on dataset
    if "comp0" in filename_data or "comp90" in filename_data:
        param_displacement = "pa1"
        param_force = "pa0"
    elif "static" in filename_data:
        param_displacement = "pa0"
        param_force = "pa1"
    else:
        raise Exception("Cannot identify parameters for given dataset type")

    # Find interpolator for displacement and force over time
    force_displacement_data = pridb.read_parametric()[["time", "pa0", "pa1"]]
    displacement_fn = interp1d(
        force_displacement_data["time"], force_displacement_data[param_displacement]
    )
    force_fn = interp1d(force_displacement_data["time"], force_displacement_data[param_force])

    # Find times corresponding to indexes of example
    metadata = pridb.read_hits()[["time", "trai"]].set_index("trai")
    try:
        metadata = metadata.iloc[idx]
    except IndexError:
        print(f"Indexes from {filename_data} were not found in {filename_metadata}")
        sys.exit(-1)

    # Interpolate displacement and force for each example
    metadata["displacement"] = displacement_fn(metadata["time"])
    metadata["force"] = force_fn(metadata["time"])
    metadata.reset_index(inplace=True, drop=True)

    metadata.to_pickle(filename_out)
    print(f"Saved metadata of {metadata.shape[0]} example to {filename_out}")
