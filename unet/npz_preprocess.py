# npz_preprocess.py

import os
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom


def delete_bad_image(data_path, txt_save_path):
    fbad = open(os.path.join(txt_save_path, 'bad.txt'), 'w')
    os.makedirs(data_path, exist_ok=True)

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".npz"):
                file_path = os.path.join(root, file)
                with np.load(file_path) as data:
                    label = data['label']
                if not label.any():
                    name = os.path.basename(os.path.normpath(file)) + '\n'
                    fbad.write(name)
                    os.remove(file_path)


class Zoom(object):

    def __init__(self, output_size: List[int]):
        self.output_size = output_size

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """缩放 image 和 label"""
        image, label = sample['image'], sample['label']
        x, y, _ = image.shape

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label}
        return sample


class RandomGenerator(object):

    def __init__(self, output_size: List[int]):
        self.output_size = output_size
        self.size_zoom = Zoom(self.output_size)

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """对 sample 进行随机增强和归一化处理"""
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = self.random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = self.random_rotate(image, label)

        sample = {'image': image, 'label': label}
        sample = self.size_zoom(sample)

        sample['image'] = sample['image'] / 255.0   # 归一化
        return sample

    def random_rot_flip(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """随机 90° 旋转（rotate）和翻转（flip）"""
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)

        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def random_rotate(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """随机角度旋转（rotate）"""
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label
