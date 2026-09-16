import csv
from pathlib import Path
import yaml


import matplotlib.pyplot as plt


def main():
    with open("./config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    output_dir = Path("outputs") / config["model"]["name"]
    with open(output_dir / "metrics.csv", "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    epochs = [int(row["epoch"]) for row in rows]

    panels = [(), (), ()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (title, columns) in zip(axes, panels):
        for column in columns:
            values = [float(row[column]) for row in rows]
            ax.plot(epochs, values, label=column, linewidth=2, marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)

    fig.suptitle(f"{config["model"]["name"]} | Kvasir-seg")
    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
