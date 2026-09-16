import argparse
import csv
from pathlib import Path


import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam


from utils.dataset import SelfDefineDataset
from utils.metrics import evaluate
from models import build_model
from utils.plot import plot_history


def main():

    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    with open("./config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default=config["datasets"]["root"], help="数据集根目录"
    )
    args = parser.parse_args()

    data_config = config["datasets"]
    data_config["root"] = args.root
    print("使用的数据集目录为：", data_config["root"])

    torch.manual_seed(config["training"]["seed"])

    dataset = SelfDefineDataset(
        data_config["root"],
        images=data_config["images"],
        masks=data_config["masks"],
        image_size=data_config["image_size"],
    )

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        data_config["split_ratio"],
        generator=torch.Generator().manual_seed(config["training"]["seed"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    print("训练集数量：", len(train_dataset))
    print("验证集数量：", len(val_dataset))
    print("测试集数量：", len(test_dataset))

    model = build_model(config["model"]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

    output_dir = Path("outputs") / config["model"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")  # 无限大
    epochs = config["training"]["epochs"]

    history = []

    # 开始训练
    for epoch in range(epochs):
        print(f"开始 Epoch {epoch+1}/{epochs}", flush=True)

        model.train()
        train_loss_sum = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
        train_loss = train_loss_sum / len(train_dataset)

        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"IoU: {val_metrics["iou"]:.4f} | F1: {val_metrics["f1"]:.4f} Precision: {val_metrics["precision"]:.4f} Recall: {val_metrics["recall"]:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print("已保存最佳模型", flush=True)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_metrics["iou"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }
        )
        with open(
            output_dir / "metrics.csv", "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    plot_history(output_dir, title=f"{config["model"]["name"]} | {config["dataset"]}")


if __name__ == "__main__":
    main()
