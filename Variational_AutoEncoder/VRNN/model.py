import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Variational_AutoEncoder.models.misc import ScatteringNet
from Variational_AutoEncoder.utils.data_utils import plot_scattering
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

# EPS = torch.finfo(torch.float).eps # numerical logs
EPS = 1e-3

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

@dataclass
class VrnnForward:
    rec_loss: torch.Tensor = None
    kld_loss: float = None
    nll_loss: float = None
    kld_values: List[torch.Tensor] = None
    encoder_mean: List[torch.Tensor] = None
    encoder_std: List[torch.Tensor] = None
    decoder_mean: List[torch.Tensor] = None
    decoder_std: List[torch.Tensor] = None
    Sx: torch.Tensor = None
    Sx_meta: dict = None
    z_latent: List[torch.Tensor] = None


class CustomTanhSim(nn.Module):
    def __init__(self):
        super(CustomTanhSim, self).__init__()

    def forward(self, x):
        # Apply the Tanh activation
        x = torch.tanh(x)
        # Scale the output to the range [-5, +5]
        x = x * 5
        return x


class CustomTanh(nn.Module):
    def __init__(self, min_val=-5, max_val=5):
        super(CustomTanh, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        # Apply Tanh
        x = torch.tanh(x)
        # Scale to the new range
        x = x * (self.max_val - self.min_val) / 2 + (self.max_val + self.min_val) / 2
        return x


class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, log_stat=None, bias=False):
        super(VRNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.x_mean = log_stat[0][1:13]
        self.x_std = log_stat[1][1:13]
        self.st0_mean = 140.37047
        self.st0_std = 18.81198

        self.transform = ScatteringNet(J=11, Q=1, T=(2 ** (11 - 7)), shape=2400)

        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim)
        )

        #recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)


    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_z_t = []
        all_kld = []
        kld_loss = 0
        nll_loss = 0

        # Convert numpy arrays to PyTorch tensors and specify dtype
        x_mean_tensor = torch.tensor(self.x_mean, dtype=torch.float32)
        x_std_tensor = torch.tensor(self.x_std, dtype=torch.float32)

        x_mean_reshaped = x_mean_tensor.reshape((1, 1, 12, 1)).to(device)
        x_std_reshaped = x_std_tensor.reshape((1, 1, 12, 1)).to(device)

        # scattering transform preprocess ------------------------------------------------------------------------------
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
        scattering_original = x
        # x = x.squeeze().transpose(0, 1)
        # scattering transform preprocess ------------------------------------------------------------------------------
        # shape of x before entering the loop: (time_index, Batch, x_t_dim) where x_t_dim is the dimension \
        # of x at each time sample
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=device)
        for t in range(x.size(0)):

            phi_x_t = self.phi_x(x[t])

            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # computing losses
            kld_value = self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            kld_loss += kld_value
            # nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            all_kld.append(kld_value)
            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)
            all_z_t.append(z_t)

        reconstructed = torch.stack(all_dec_mean, dim=0)
        rec_loss = F.mse_loss(reconstructed, x, reduction='mean')
        # results = {
        #     "rec_loss": rec_loss,
        #     "kld_loss": kld_loss,
        #     "nll_loss": nll_loss,
        #     "encoder_mean": all_enc_mean,
        #     "encoder_std": all_enc_std,
        #     "decoder_mean": all_dec_mean,
        #     "decoder_std": all_dec_std,
        #     "kld_values": all_kld,
        #     "Sx": scattering_original,
        #     "Sx_meta": meta,
        #     "z_latent": all_z_t
        # }
        results = VrnnForward(
            rec_loss=rec_loss,
            kld_loss=kld_loss,
            nll_loss=nll_loss,
            encoder_mean=all_enc_mean,
            encoder_std=all_enc_std,
            decoder_mean=all_dec_mean,
            decoder_std=all_dec_std,
            kld_values=all_kld,
            Sx=scattering_original,
            Sx_meta=meta,
            z_latent=all_z_t
        )
        return results
        # return rec_loss, kld_loss, nll_loss, \
        #     (all_enc_mean, all_enc_std), \
        #     (all_dec_mean, all_dec_std), (scattering_original, meta), all_z_t


    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim, device=device)

        h = torch.zeros(self.n_layers, 1, self.h_dim, device=device)
        for t in range(seq_len):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    # def _reparameterized_sample(self, mean, std):
    #     """using std to sample"""
    #     eps = torch.empty(size=std.size(), device=device, dtype=torch.float).normal_()
    #     return eps.mul(std).add_(mean)



    def _reparameterized_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        std_2 = torch.clamp(std_2, min=1e-9)
        std_1 = torch.clamp(std_1, min=1e-9)

        kld_element = (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def _nll_bernoulli(self, theta, x):
        theta = torch.clamp(theta, min=1e-9)
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log((1-theta) + EPS))

    # def _nll_gauss(self, mean, std, x):
    #     return torch.sum(torch.log(std + EPS) + torch.log(2 * torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))

    def _nll_gauss(self, mean, std, x):
        pi_tensor = torch.tensor(2 * torch.pi, device=std.device, dtype=std.dtype)  # Convert to tensor
        return torch.sum(torch.log(std + EPS) + torch.log(pi_tensor) / 2 + (x - mean).pow(2) / (2 * std.pow(2)))
