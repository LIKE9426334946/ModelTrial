import torch
import yaml

from torch.utils.data import DataLoader, random_split


from utils.dataset import SelfDefineDataset


def main():
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    with open("./config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["datasets"]
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

    # 读取一个batch
    images, masks = next(iter(train_loader))
    print("图片形状：", images.shape)
    print("标签形状：", masks.shape)
    print("图片类型：", images.dtype)
    print("标签类型：", masks.dtype)
    print("图片范围：", images.min().item(), images.max().item())
    print("标签取值：", torch.unique(masks))


if __name__ == "__main__":
    main()
