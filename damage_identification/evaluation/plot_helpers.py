import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib


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
        pad_inches=0.25,
    )

    plt.close()


def format_plot(
    xlocator=matplotlib.ticker.AutoMinorLocator(),
    ylocator=matplotlib.ticker.AutoMinorLocator(),

):
    fig = plt.gcf()
    for ax in fig.axes[:1]:

        ax.get_xaxis().set_minor_locator(xlocator)
        ax.get_yaxis().set_minor_locator(ylocator)

        ax.grid(b=True, which="major", linewidth=1.0)
        ax.grid(b=True, which="minor", linewidth=0.5, linestyle="-.")

    fig.tight_layout(pad=2.5, h_pad=0.3, w_pad=0.2)



def format_plot_2D(
    xlocator=matplotlib.ticker.AutoMinorLocator(),
    ylocator=matplotlib.ticker.AutoMinorLocator(),
    zeroline=True,
):
    fig = plt.gcf()
    for ax in fig.axes[:1]:
        if zeroline:
            ax.axhline(0, linewidth=1.5, c="black")

        ax.get_xaxis().set_minor_locator(xlocator)
        ax.get_yaxis().set_minor_locator(ylocator)
        ax.set_ylim(ymin=0)
        ax.grid(b=True, which="major", linewidth=1.0)
        ax.grid(b=True, which="minor", linewidth=0.5, linestyle="-.")

    fig.tight_layout(pad=0.1, h_pad=0.4, w_pad=0.4)