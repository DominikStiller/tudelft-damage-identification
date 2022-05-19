import pandas as pd
import matplotlib.pyplot as plt
from damage_identification.evaluation.plot_helpers import save_plot, format_plot_2d_linear


def force_displacement(filename, results_folder: str):
    data = pd.read_csv(
        f"../../data/{filename}", delimiter="\t", encoding="utf-8", header=2, on_bad_lines="skip"
    )
    data = data.drop(index=data.index[data["Time"].str.isalpha()])
    colreplacement = {
        ".* kN MTS Force": "force",
        "Running Time": "runtime",
        "Time": "time",
        ".* MTS Displacement": "displacement",
    }
    data.columns = data.columns.to_series().replace(colreplacement, regex=True)

    for column in data.columns:
        data[column] = [x.replace(",", ".") for x in data[column]]
        data[column] = data[column].astype(float)

    data["force"], data["displacement"] = -data["force"], -data["displacement"]

    plt.plot(data["displacement"], data["force"])
    plt.xlabel("Displacement [mm]")
    plt.ylabel("Force [kN]")

    format_plot_2d_linear()
    save_plot(results_folder, name=f"force_displacement_{filename}", fig=plt)
