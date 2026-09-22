import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from datasets.kvasir_dataset import SelfDefineDataset
from utils.metrics import evaluate
from models import build_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("./config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_name = config["dataset"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=config["datasets"][dataset_name]["root"],
        help="数据集根目录",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config["model"]["name"],
        help="选择模型",
    )
    args = parser.parse_args()
    config["model"]["name"] = args.model

    data_config = config["datasets"][dataset_name]
    data_config["root"] = args.root
    print("使用的数据集目录为：", data_config["root"])

    dataset = SelfDefineDataset(
        root=data_config["root"],
        images=data_config["images"],
        masks=data_config["masks"],
        image_size=data_config["image_size"],
    )
    _, _, test_dataset = random_split(
        dataset,
        data_config["split_ratio"],
        generator=torch.Generator().manual_seed(config["training"]["seed"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    output_dir = Path("outputs") / config["model"]["name"]
    model = build_model(config["model"]).to(device)
    print(f"使用的模型为{config["model"]["name"]}")

    state_dict = torch.load(
        output_dir / "best_model.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)

    # 评估整个测试集
    criterion = nn.BCEWithLogitsLoss()
    results = evaluate(model, test_loader, criterion, device)
    print("测试集样本数：", len(test_dataset))
    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    images, masks = next(iter(test_loader))
    with torch.no_grad():
        probabilities = model(images.to(device)).sigmoid().cpu()

    predictions = (probabilities >= 0.5).float()
    count = min(4, images.size(0))
    fig, axes = plt.subplots(count, 3, figsize=(10, count * 3), squeeze=False)

    for i in range(count):
        axes[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Image")

        axes[i, 1].imshow(masks[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(predictions[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction")

        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_dir / "predictions.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()
