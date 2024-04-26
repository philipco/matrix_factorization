from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io
import os
import torch


class CelebaDataset(Dataset):
    """From https://www.kaggle.com/code/ekansh/celeb-vae."""
    def __init__(self, data_dir, partition_file_path, identity_file_path, split, transform):
        self.partition_file = pd.read_csv(partition_file_path)
        self.identity = pd.read_csv(identity_file_path, sep=" ", header=None, names=["Image", "Identity"])
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

    def __len__(self):
        self.partition_file_sub = self.partition_file[self.partition_file["partition"].isin(self.split)]
        return len(self.partition_file_sub)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir,
                                self.partition_file.iloc[idx, 0])
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem_by_identity__(self, idx):
        indices = self.identity.index[self.identity['Identity'] == idx].tolist()
        return torch.stack([torch.cat([self.__getitem__(i)[j] for j in range(3)], dim=1).view(-1) for i in indices]).numpy()