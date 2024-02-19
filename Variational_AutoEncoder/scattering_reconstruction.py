import torch
from torch import nn
from Variational_AutoEncoder.models.encoder import LSTMEncoder, ConvLinEncoder
from Variational_AutoEncoder.models.decoder import LSTMDecoder, ConvLinDecoder
import numpy as np
from torch.autograd import backward
import matplotlib.pyplot as plt
from kymatio.torch import Scattering1D
import os
import yaml
import logging
from datetime import datetime
import sys
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
from datasets.custom_datasets import JsonDatasetPreload
from Variational_AutoEncoder.models.model import VAE_linear
import matplotlib.pyplot as plt
import numpy as np
from Variational_AutoEncoder.utils.data_utils import plot_original_reconstructed
import os
from models.misc import VAELoss
from utils.data_utils import plot_scattering
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ScatteringNet(nn.Module):
    def __init__(self, J, Q, T, shape):
        super(ScatteringNet, self).__init__()
        self.scat = Scattering1D(J=J, Q=Q, T=T, shape=shape)

    def forward(self, x):
        # x = x.permute(0, 2, 1)  # Equivalent to Permute in TensorFlow
        # x_tensor = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(2)
        x = x.permute(0, 2, 1)

        x = x.contiguous()
        return self.scat(x)

    def meta(self):
        return self.scat.meta()



class ScatteringReconstructionNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ScatteringReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_harmonic_signal(T, num_intervals=4, gamma=0.9, random_state=42):
    """
    Generates a harmonic signal, which is made of piecewise constant notes
    (of random fundamental frequency), with half overlap
    """
    rng = np.random.RandomState(random_state)
    num_notes = 2 * (num_intervals - 1) + 1
    support = T // num_intervals
    half_support = support // 2

    base_freq = 0.1 * rng.rand(num_notes) + 0.05
    phase = 2 * np.pi * rng.rand(num_notes)
    window = np.hanning(support)
    x = np.zeros(T, dtype='float32')
    t = np.arange(0, support)
    u = 2 * np.pi * t
    for i in range(num_notes):
        ind_start = i * half_support
        note = np.zeros(support)
        for k in range(1):
            note += (np.power(gamma, k) *
                     np.cos(u * (k + 1) * base_freq[i] + phase[i]))
        x[ind_start:ind_start + support] += note * window

    return x


if __name__ == "__main__":
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    config_file_path = r'config_arguments.yaml'
    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Preparing training and testing datasets --------------------------------------------------------------------------
    dataset_dir = os.path.normpath(config['dataset_config']['dataset_dir'])
    stat_path = os.path.normpath(config['dataset_config']['stat_path'])
    batch_size = 64
    fhr_healthy_dataset = JsonDatasetPreload(dataset_dir)
    dataset_size = len(fhr_healthy_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = random_split(fhr_healthy_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    one_fhr = train_dataset[19]

    # transform --------------------------------------------------------------------------------------------------------
    scattering1D_tr = ScatteringNet(J=11, Q=1, T=(2 ** (11 - 7)), shape=2400)
    one_fhr = one_fhr.view(1, len(one_fhr))
    [Sx, Px] = scattering1D_tr(one_fhr)
    meta = scattering1D_tr.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)
    combined_orders = np.where((meta['order'] == 0) | (meta['order'] == 1))
    selected_orders = torch.from_numpy(combined_orders[0])
    x = Sx[:, :, selected_orders, :]
    x = x.squeeze(1)
    x_scattering = x.squeeze(1)
    plot_scattering(signal=one_fhr.permute(1, 0), Sx=x_scattering.squeeze(0), do_plot_rec=False)

    x_random = torch.randn((1, 2400))

    # fhr_hie_dataset = FHRDataset(hie_list[0:10])
    # hie_dataloader = DataLoader(fhr_hie_dataset, batch_size=1, shuffle=False)


    # T = 2 ** 13
    # x = torch.from_numpy(generate_harmonic_signal(T))
    # plt.figure(figsize=(8, 2))
    # plt.plot(x.numpy())
    # plt.title("Original signal")
    #
    # plt.figure(figsize=(8, 8))
    # plt.specgram(x.numpy(), Fs=1024)
    # plt.title("Spectrogram of original signal")
    #
    # plt.show()
    #
    # J = 6
    # Q = 16
    # sc_transform = ScatteringNet(J=11, Q=1, T=(2 ** (11 - 7)), shape=8192)
    #
    # x = x.to(device)
    # x = x.view(1, len(x))
    #
    # [Sx, Px] = sc_transform(x)
    #
    # learning_rate = 100
    # bold_driver_accelerator = 1.1
    # bold_driver_brake = 0.55
    # n_iterations = 200
    #
    # torch.manual_seed(0)
    # y = torch.randn((T,), requires_grad=True, device=device)
    # y = y.view(1, len(y))
    # [Sy, Py] = sc_transform(y)
    #
    # history = []
    # signal_update = torch.zeros_like(x, device=device)
    #
    # # Iterate to recontsruct random guess to be close to target.
    # for k in range(n_iterations):
    #     # Backpropagation.
    #     err = torch.norm(Sx - Sy)
    #
    #     if k % 10 == 0:
    #         print('Iteration %3d, loss %.2f' % (k, err.detach().cpu().numpy()))
    #
    #     # Measure the new loss.
    #     history.append(err.detach().cpu())
    #
    #     backward(err)
    #
    #     delta_y = y.grad
    #
    #     # Gradient descent
    #     with torch.no_grad():
    #         signal_update = - learning_rate * delta_y
    #         new_y = y + signal_update
    #     new_y.requires_grad = True
    #
    #     # New forward propagation.
    #     Sy = sc_transform(new_y)
    #
    #     if history[k] > history[k - 1]:
    #         learning_rate *= bold_driver_brake
    #     else:
    #         learning_rate *= bold_driver_accelerator
    #         y = new_y
    #
    # plt.figure(figsize=(8, 2))
    # plt.plot(history)
    # plt.title("MSE error vs. iterations")
    #
    # plt.figure(figsize=(8, 2))
    # plt.plot(y.detach().cpu().numpy())
    # plt.title("Reconstructed signal")
    #
    # plt.figure(figsize=(8, 8))
    # plt.specgram(y.detach().cpu().numpy(), Fs=1024)
    # plt.title("Spectrogram of reconstructed signal")
    # plt.show()
