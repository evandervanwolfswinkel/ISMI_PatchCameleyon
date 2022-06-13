import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
# import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.x = self.hdf5_to_tensor(x_path, "x", imgs=True)
        self.y = self.hdf5_to_tensor(y_path, "y")
        self.transform = transform

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item][0][0]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.x)  # "Length X: {}, Length Y: {}".format(len(self.x), len(self.y))

    def hdf5_to_tensor(self, path, idx, imgs=False):
        if imgs:
            return torch.from_numpy(np.array(h5py.File(path, "r")[idx])).permute(0, 3, 1, 2).float()
        else:
            return torch.from_numpy(np.array(h5py.File(path, "r")[idx])).float()

# class CustomLazyDataset(Dataset):
#     def __init__(self, x_path, y_path, transform=None):
#         self.data = pd.read_csv()
#         self.x = self.hdf5_to_tensor(x_path, "x", imgs=True)
#         self.y = self.hdf5_to_tensor(y_path, "y")
#         self.transform = transform

#     def __getitem__(self, item):
#         x = self.x[item]/255.
#         y = self.y[item][0][0]
#         x = self.transform(x)
#         return x, y

#     def __len__(self):
#         return len(self.x)  # "Length X: {}, Length Y: {}".format(len(self.x), len(self.y))

#     def hdf5_to_tensor(self, path, idx, imgs=False):
#         if imgs:
#             return torch.from_numpy(np.array(h5py.File(path, "r")[idx])).permute(0, 3, 1, 2).float()
#         else:
#             return torch.from_numpy(np.array(h5py.File(path, "r")[idx])).float()


# if __name__ == '__main__':
#     dataset = CustomDataset("/Users/joshuakoopmans/Downloads/camelyonpatch_level_2_split_test_x.h5",
#                             "/Users/joshuakoopmans/Downloads/camelyonpatch_level_2_split_test_y.h5")
#     test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
#
#     for epoch in range(2):
#         for batch_ndx, (x, y) in enumerate(test_loader):
#             print(batch_ndx, x.shape, y.shape)
