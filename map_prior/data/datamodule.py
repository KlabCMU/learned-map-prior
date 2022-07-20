import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import pandas as pd
import tqdm
import glob

# our library imports
import os
import sys
from .map_processing import *
from .imu_processing import *

# deep learning imports
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

import pytorch_lightning as pl

from .dataset import *


class MapZoomedDataModule(pl.LightningDataModule):

    def __init__(self,
                 train_building: str,
                 train_files: list,
                 val_files: list,
                 batch_size: int = 32,
                 kernel_len: int = 5,
                 stride: int = 1,
                 samples_per_second: int = 100,
                 transform=False,
                 infeasible_space_weight=None,
                 ):
        super().__init__()
        self.train_building = train_building
        self.batch_size = batch_size
        self.train_files = train_files
        self.val_files = val_files
        self.kernel_len = kernel_len
        self.stride = stride
        self.samples_per_second = samples_per_second
        self.transform = transform
        self.max_size = 0
        self.infeasible_space_weight = infeasible_space_weight
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        self.train_map = get_map_by_name(self.train_building)
        print(f"training map size: {self.train_map.shape}")

    def proc_file_train(self, filename, stride):
        data = np.load(filename, allow_pickle=True).item()

        # df.loc[:, ['aligned_posX', 'aligned_posY']].values
        positions = data["GT"]
        image_positions = world_to_image_coords(
            positions, self.train_building)
        windowed_GT_positions = window_data(self.kernel_len*self.samples_per_second,
                                            stride*self.samples_per_second,
                                            image_positions)
        windowed_pred_positions = windowed_GT_positions.copy()
        return {"GT": windowed_GT_positions, "pred": windowed_pred_positions}

    def batch_bbox_size_np(self, points):
        return np.array([np.max(points[..., 0], axis=1) - np.min(points[..., 0], axis=1)+1,
                         np.max(points[..., 1], axis=1) - np.min(points[..., 1], axis=1)+1]).T

    def setup(self, stage: str = None):

        all_train_samples = []
        for file in self.train_files:
            window_pos = self.proc_file_train(file, stride=self.stride)
            all_train_samples.append(window_pos)
        all_train_samples = {"GT": np.concatenate([a["GT"] for a in all_train_samples], axis=0),
                             "pred": np.concatenate([a["pred"] for a in all_train_samples], axis=0)}

        all_val_samples = []
        for file in self.val_files:
            window_pos = self.proc_file_train(file, stride=self.stride)
            all_val_samples.append(window_pos)
        all_val_samples = {"GT": np.concatenate([a["GT"] for a in all_val_samples], axis=0),
                           "pred": np.concatenate([a["pred"] for a in all_val_samples], axis=0)}

        all_samples = np.concatenate(
            (all_train_samples["GT"], all_val_samples["GT"]), axis=0)
        bbox_sizes = self.batch_bbox_size_np(all_samples)
        max_bbox_size = np.max(bbox_sizes, axis=0)

        # this padding is for small-UNet, and would need to be changed for e.g. a big unet (more downsamples, needs to be divisible by more powers of two)
        n_downsamples = 3
        divisible = 2**n_downsamples
        max_bbox_size[0] += divisible - max_bbox_size[0] % divisible
        max_bbox_size[1] += divisible - max_bbox_size[1] % divisible

        # make square, so we can rotate more easily
        max_bbox_size[0] = np.max(max_bbox_size)
        max_bbox_size[1] = np.max(max_bbox_size)
        max_bbox_size *= 2  # increase size to introduce some randomness, and to provide enough space to augment the trajectories
        print(f"map segment size: {max_bbox_size}")
        self.max_size = max_bbox_size/2

        self.train_dataset = MapPaddedDataset(all_train_samples, self.train_map,
                                              map_segment_size=max_bbox_size,
                                              transform=self.transform, rotate=True, infeasible_space_weight=self.infeasible_space_weight)

        self.val_dataset = MapPaddedDataset(all_val_samples, self.train_map,
                                            map_segment_size=max_bbox_size, infeasible_space_weight=self.infeasible_space_weight,
                                            transform=self.transform)

        self.all_train_samples = all_train_samples
        self.all_val_samples = all_val_samples

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=24)


if __name__ == "__main__":
    dm = MapZoomedDataModule(data_dir="../../datasets/IDOL_dataset/",
                             batch_size=32, kernel_len=5, transform=True)
    dm.prepare_data()
    print("------")
    dm.setup()
    print(dm.max_size)
    dl = dm.train_dataloader()
    for i_batch, sample_batched in enumerate(tqdm.tqdm(dl)):
        normalized_traj = sample_batched['normalized input trajectory']
