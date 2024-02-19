from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import json


class FHRDataset(Dataset):
    # todo how to get the GUID
    def __init__(self, list_dicts):
        # Concatenating 'fhr' from both lists
        self.data = [d['fhr'] for d in list_dicts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert data to PyTorch tensor
        sample = torch.tensor(self.data[idx], dtype=torch.float)
        return sample


#  Dataset class from numpy .npy data
class SignalDataset(Dataset):
    def __init__(self, data):
        # Assuming data is a NumPy array of shape (4000, 300)
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sample at the specified index
        sample = self.data[idx]
        return sample


class JsonDatasetPreload(Dataset):
    def __init__(self, json_folder_path, mean=None, std=None):
        self.data_files = [os.path.join(json_folder_path, file) for file in os.listdir(json_folder_path) if
                           file.endswith('.json')]
        self.samples = []
        self.mean = mean
        self.std = std

        # Load data
        for file_path in self.data_files:
            with open(file_path, 'r') as file:
                data = json.load(file)
                # Assuming each file contains a single sample for simplicity
                self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # todo check how to normalize fhr + UP
        sample_data = self.samples[idx]
        # Extracting the `fhr` data and possibly other information
        fhr = torch.tensor(sample_data['fhr'])
        up = torch.tensor(sample_data['up'])
        target = torch.tensor(sample_data['target'])
        sample_weight = torch.tensor(sample_data['sample_weights'])
        # return fhr, target, sample_weight
        if self.mean is not None and self.std is not None:
            # fhr = (fhr - self.mean.view(1, -1)) / self.std.view(1, -1) # for fhr + up
            fhr = (fhr - self.mean) / self.std
        return fhr


class JsonDataset(Dataset):
    def __init__(self, json_folder_path):
        self.data_files = [os.path.join(json_folder_path, file) for file in os.listdir(json_folder_path) if
                           file.endswith('.json')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load only the needed JSON file
        file_path = self.data_files[idx]
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract data
        fhr = torch.tensor(data['fhr'])
        up = torch.tensor(data['up'])
        target = torch.tensor(data['target'])
        sample_weight = torch.tensor(data['sample_weights'])

        return fhr, target, sample_weight
