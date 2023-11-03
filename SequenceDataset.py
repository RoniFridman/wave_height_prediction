import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length):
        self.df = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        self.columns_mean = {}
        self.columns_std = {}

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

    def normalize_features_and_target(self, mean_std_dicts=None):
        if mean_std_dicts is None:
            for c in self.df.columns:
                self.columns_mean[c] = self.df[c].mean()
                self.columns_std[c] = self.df[c].std()
                self.df[c] = (self.df[c] - self.columns_mean[c]) / self.columns_std[c]
        else:
            mean_dict, std_dict = mean_std_dicts
            self.columns_mean = mean_dict if self.columns_mean == {} else self.columns_mean
            self.columns_std = std_dict if self.columns_std == {} else self.columns_std
            for c in self.df.columns:
                self.df[c] = (self.df[c] - mean_dict[c]) / std_dict[c]
        self.y = torch.tensor(self.df[self.target].values).float()
        self.X = torch.tensor(self.df[self.features].values).float()
