import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from utils.config import get_output_dir, load_config
from utils.task import build_criterion, predict_classes
from datasets.kvasir_dataset import SelfDefineDataset
from utils.metrics import evaluate
from models import build_model
from datasets import build_dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config()
    dataset_name = config["dataset"]
    data_config = config["datasets"][dataset_name]
    print("使用的数据集为：", dataset_name)
    print("使用的数据集目录为：", data_config["root"])

    test_dataset = build_dataset(config, "test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    output_dir = get_output_dir(config)
    model = build_model(config["model"], data_config).to(device)
    print(f"使用的模型为{config['model']['name']}")

    state_dict = torch.load(
        output_dir / "best_model.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)

    # 评估整个测试集
    criterion = build_criterion(data_config)
    results = evaluate(model, test_loader, criterion, device, data_config)
    print("测试集样本数：", len(test_dataset))

    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    images, masks = next(iter(test_loader))
    with torch.no_grad():
        logits = model(images.to(device))
        predictions = predict_classes(logits, data_config).cpu()

    binary = data_config["mode"] == "binary"
    count = min(4, images.size(0))
    fig, axes = plt.subplots(count, 3, figsize=(10, count * 3), squeeze=False)
    cmap = "gray" if binary else plt.get_cmap("tab10", data_config["num_classes"])
    vmax = data_config["num_classes"] - 1

    for i in range(count):
        if images.size(1) == 1:
            axes[i, 0].imshow(images[i, 0].numpy(), cmap="gray")
        else:
            axes[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Image")

        mask = masks[i, 0] if binary else masks[i]
        prediction = predictions[i, 0] if binary else predictions[i]
        axes[i, 1].imshow(
            mask.numpy(), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest"
        )
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(
            prediction.numpy(), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest"
        )
        axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_dir / "predictions.png", dpi=100)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
