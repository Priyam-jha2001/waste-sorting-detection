"""One small helper for the bar charts used in Tasks 1-3."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # draw to files, no window needed
import matplotlib.pyplot as plt          # noqa: E402

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})


def bars(ax, labels: list[str], series: dict[str, list[float]], title: str, fmt: str = "{:.2f}") -> None:
    """Grouped bar chart with the value written on each bar."""
    width = 0.8 / len(series)
    for i, (name, values) in enumerate(series.items()):
        x = [k + (i - (len(series) - 1) / 2) * width for k in range(len(labels))]
        for rect, v in zip(ax.bar(x, values, width, label=name), values):
            ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(fontsize=8)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
