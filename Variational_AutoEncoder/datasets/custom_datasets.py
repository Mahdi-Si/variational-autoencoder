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
    def __init__(self, json_folder_path):
        self.data_files = [os.path.join(json_folder_path, file) for file in os.listdir(json_folder_path) if
                           file.endswith('.json')]
        self.samples = []
        for file_path in self.data_files:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_data = self.samples[idx]
        fhr = torch.tensor(sample_data['fhr'])
        up = torch.tensor(sample_data['up'])
        target = torch.tensor(sample_data['target'])
        sample_weight = torch.tensor(sample_data['sample_weights'])
        # return fhr, target, sample_weight
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
