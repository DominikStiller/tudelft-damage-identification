import pandas as pd
import matplotlib.pyplot as plt
from damage_identification.evaluation.plot_helpers import save_plot, format_plot_2d_linear


def force_displacement(data: pd.DataFrame, filename: str, results_folder: str):

    plt.plot(data["displacement"], data["force"])
    plt.xlabel("Displacement [mm]")
    plt.ylabel("Force [kN]")

    format_plot_2d_linear()
    save_plot(results_folder=results_folder, name=f"force_displacement_{filename}", fig=plt)
