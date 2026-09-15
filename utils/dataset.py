import yaml

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# 定义数据集类
class SelfDefineDataset(Dataset):
    def __init__(self, root, images, masks):
        super().__init__()
        

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)
