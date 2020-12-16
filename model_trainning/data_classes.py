from __future__ import print_function
import os
import pandas as pd  # For easier csv parsingFor easier csv parsing
import numpy as np
import torch


class DataSetClass(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, Data_Path, window_len=None,include_Vy=False,  transform=None):
        """
        Args:
            Data_Path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.data = pd.read_csv(csv_file,skiprows=[0])
        self.data_goal = pd.read_csv(Data_Path+"goals.csv")
        self.data_range = pd.read_csv(Data_Path+"ranges.csv")
        self.data_cmd = pd.read_csv(Data_Path+"command_velocities.csv")
        self.data_size = len(self.data_goal)
        self.include_Vy = include_Vy

        self.transform = transform
        self.window = window_len
        if window_len is not None:
            self.data_upsampled = self.getWScannedData()
            self.data_size = len(self.data_upsampled)
    # __len__ so that len(dataset) returns the size of the dataset.

    def __len__(self):
        return self.data_size

    # __getitem__`` to support the indexing such that ``dataset[i]`` can
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.window is not None:
            idx = self.data_upsampled[idx]
        # cmd
        if self.include_Vy:
          steering_cmd = self.data_cmd.iloc[idx,[0,2]]
        else:
          steering_cmd = self.data_cmd.iloc[idx]
        steering_cmd = np.asarray(steering_cmd).astype("float")
        # goal
        goal_pose = self.data_goal.iloc[idx]
        goal_pose = np.asarray(goal_pose).astype("float")
        # scan
        scan = self.data_range.iloc[idx]
        scan = np.asarray(scan).astype("float")
        scan = np.expand_dims(scan, axis=0)

        sample = {'scan': scan, 'goal': goal_pose, 'steering': steering_cmd}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def getWScannedData(self):
        window = []
        data_idx_upsampled = []
        for i in range(self.data_size):
            if i < self.window:
                window.append(i)
                data_idx_upsampled.append(i)
                continue

            window = window[1:]
            window.append(i)
            data_idx_upsampled.extend(window)

        return data_idx_upsampled

# Transformations


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        scan, goal, steering = sample["scan"], sample["goal"], sample["steering"]

        return {'scan': torch.from_numpy(scan),
                'goal': torch.from_numpy(goal),
                'steering': torch.from_numpy(steering)}


class Normalize(object):

    def __call__(self, sample):
        scan, steering, goal = sample['scan'], sample['steering'], sample["goal"]
        # normalize scans
        mean = np.array([0.7932440954039575])
        variance = np.array([1.4043073858035697])
        mean = torch.from_numpy(mean)
        variance = torch.from_numpy(variance)
        scan = (scan - mean)/variance

        # normalize steering
        mean = np.array([1.82032557e-01, 2.72741526e-03])
        variance = np.array([0.07903122, 0.41434146])
        mean = torch.from_numpy(mean)
        variance = torch.from_numpy(variance)
        steering = (steering - mean)/variance

        # normalize the goal
        mean = np.array([1.50223488,  0.01608201, -0.01331008])
        variance = np.array([0.96882731, 0.88992184, 1.34744442])
        mean = torch.from_numpy(mean)
        variance = torch.from_numpy(variance)
        goal = (goal - mean) / variance

        return {'scan': scan, 'steering': steering, 'goal': goal}
