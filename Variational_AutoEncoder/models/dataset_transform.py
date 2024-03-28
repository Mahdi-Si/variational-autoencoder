import torch
import torch.nn as nn
import numpy as np
from Variational_AutoEncoder.models.misc import ScatteringNet

class ScatteringTransform(nn.Module):
    def __init__(self, input_size=None, input_dim=None, log_stat=None, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_mean = log_stat[0][1:13]
        self.x_std = log_stat[1][1:13]
        self.st0_mean = 140.37047
        self.st0_std = 18.81198
        self.input_size = input_size  # size of the input signal
        self.input_dim = input_dim  # dimension of the input signal
        self.transform = ScatteringNet(J=11, Q=1, T=(2 ** (11 - 7)), shape=input_size)
        self.device = device

    def forward(self, x):
        x_mean_tensor = torch.tensor(self.x_mean, dtype=torch.float32)
        x_std_tensor = torch.tensor(self.x_std, dtype=torch.float32)

        x_mean_reshaped = x_mean_tensor.reshape((1, 1, 12, 1)).to(self.device)
        x_std_reshaped = x_std_tensor.reshape((1, 1, 12, 1)).to(self.device)

        [Sx, Px] = self.transform(x)  # Sx shape(64, 1, 76, 300)
        meta = self.transform.meta()
        order0 = np.where(meta['order'] == 0)
        order1 = np.where(meta['order'] == 1)
        order2 = np.where(meta['order'] == 2)
        combined_orders = np.where((meta['order'] == 0) | (meta['order'] == 1))
        selected_orders = torch.from_numpy(combined_orders[0])
        selected_orders_t0 = torch.from_numpy(order0[0])
        selected_orders_t1 = torch.from_numpy(order1[0])
        # x = Sx[:, :, selected_orders, :]
        x_t0 = Sx[:, :, selected_orders_t0, :]
        x_t0_normalized = (x_t0 - self.st0_mean) / self.st0_std
        x_t1 = Sx[:, :, selected_orders_t1, :]
        x_t1_normalized = (torch.log(x_t1 + 1e-4) - x_mean_reshaped) / x_std_reshaped
        x = torch.cat((x_t0_normalized, x_t1_normalized), dim=2)
        # x = x.squeeze(1).permute(0, 2, 1)  # (batch_size, 300, 13)
        x = x.squeeze(1).permute(2, 0, 1)
        x = x[:, :, 0:self.input_dim]  # x-> (input_size, batch, input_dim)
        return x, meta
