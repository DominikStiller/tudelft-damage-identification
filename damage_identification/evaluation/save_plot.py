import datetime
import os
import matplotlib.pyplot as plt


def save_plot(name: str, fig, type="pdf"):
    current_time = datetime.datetime.now()
    time = current_time.strftime("%Y-%m-%d_%H-%M")
    os.makedirs(
        f"data/plots/{time}",
        exist_ok=True,
    )
    fig.savefig(
        f"data/plots/{time}/{name}.{type}",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()
