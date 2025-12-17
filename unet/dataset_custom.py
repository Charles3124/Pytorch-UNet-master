import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import time
import datetime
from tqdm import tqdm


class Custom_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        if self.split == "testtrainset":
            self.sample_list = open(os.path.join(list_dir, "train" + '.txt')).readlines()
        elif self.split == "test_val":
            self.sample_list = open(os.path.join(list_dir, "test_vol" + '.txt')).readlines()
        else:
            self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        elif self.split == "test_vol" or self.split == "test_picked" or self.split == "testtrainset":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']

            image = torch.from_numpy(image.astype(np.float32))
            image = image.permute(2, 0, 1)
            label = torch.from_numpy(label.astype(np.float32))
        elif self.split == "test_val":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            # 修改

            # image = torch.from_numpy(image.astype(np.float32))
            # image = image.permute(2, 0, 1)
            # label = torch.from_numpy(label.astype(np.float32))
        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']

            image = torch.from_numpy(image.astype(np.float32))
            image = image.permute(2, 0, 1)
            label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample