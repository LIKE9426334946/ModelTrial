from pathlib import Path

import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam


from utils.dataset import SelfDefineDataset
from models.model_01 import Model01


def main():

    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    with open("./config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["datasets"]
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    print("训练集数量：", len(train_dataset))
    print("验证集数量：", len(val_dataset))
    print("测试集数量：", len(test_dataset))

    model = Model01(out_channels=config["model"]["out_channels"]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")  # 无限大
    epochs = config["training"]["epochs"]

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

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss_sum += loss.item() * images.size(0)
        val_loss = val_loss_sum / len(val_dataset)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print("已保存最佳模型", flush=True)


if __name__ == "__main__":
    main()
