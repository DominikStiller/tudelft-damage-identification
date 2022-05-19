import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb

# Initialize seaborn formatting once module is loaded
sb.set(
    context="paper",
    style="ticks",
    font_scale=1.6,
    font="sans-serif",
    rc={
        "lines.linewidth": 1.2,
        "axes.titleweight": "bold",
    },
)


def save_plot(results_folder: str, name: str, fig, type="pdf"):
    plots_folder = os.path.join(results_folder, "plots")
    os.makedirs(
        plots_folder,
        exist_ok=True,
    )
    fig.savefig(
        os.path.join(plots_folder, f"{name}.{type}"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close()


def format_plot_3d():
    fig = plt.gcf()
    fig.tight_layout(pad=2.5, h_pad=0.3, w_pad=0.2)


def format_plot_2d(
    xlocator=matplotlib.ticker.AutoMinorLocator(),
    ylocator=matplotlib.ticker.AutoMinorLocator(),
    zeroline=False,
):
    fig = plt.gcf()
    for ax in fig.axes:
        if zeroline:
            ax.axhline(0, linewidth=1.5, c="black")

        ax.get_xaxis().set_minor_locator(xlocator)
        ax.get_yaxis().set_minor_locator(ylocator)
        ax.grid(b=True, which="major", linewidth=1.0)
        ax.grid(b=True, which="minor", linewidth=0.5, linestyle="-.")

    fig.tight_layout(pad=0.1, h_pad=0.4, w_pad=0.4)
