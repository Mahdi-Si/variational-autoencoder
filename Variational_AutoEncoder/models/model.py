import torch
from torch import nn
from Variational_AutoEncoder.models.encoder import LSTMEncoder, ConvLinEncoder
from Variational_AutoEncoder.models.decoder import LSTMDecoder, ConvLinDecoder
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from .misc import EncoderProjection, \
    DecoderProjection, \
    ScatteringNet

class VAE(nn.Module):
    def __init__(self, input_size=None, input_dim=None, dec_hidden_dim=None, enc_hidden_dim=None,
                 latent_dim=None, latent_size=None, num_LSTM_layers=None, log_stat=None, device=None):
        super(VAE, self).__init__()
        self.x_mean = log_stat[0][1:13]
        self.x_std = log_stat[1][1:13]
        self.st0_mean = 140.37047
        self.st0_std = 18.81198
        self.device = device
        self.input_size = input_size  # size of the input signal
        self.input_dim = input_dim  # dimension of the input signal
        self.dec_hidden_dim = dec_hidden_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        self.num_LSTM_layers = num_LSTM_layers
        self.transform = ScatteringNet(J=11, Q=1, T=(2 ** (11 - 7)), shape=2400)
        self.encoder = LSTMEncoder(input_size=self.input_size, input_dim=self.input_dim,
                                   hidden_dims=[13, 10, 7],
                                   latent_size=self.latent_size, latent_dim=self.latent_dim)
        self.decoder = LSTMDecoder(latent_size=self.latent_size, latent_dim=self.latent_dim,
                                   hidden_dims=[7, 10, 13],
                                   output_dim=self.input_dim, output_size=self.input_size,
                                   num_layers=num_LSTM_layers)
        # self.enc_projection = EncoderProjection(seq_len=300, hidden_dim=256, latent_dim=128)
        self.linear_mean = nn.Linear(150, 150)
        self.linear_logvar = nn.Linear(150, 150)
        self.decoder_projection = DecoderProjection(latent_dim=128, seq_len=300)
        self.linear_decoder_init = nn.Linear(self.latent_dim, input_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        original_input = x
        # x input shape(batch_size, signal_length)
        # Scattering transform and the normalization -------------------------------------------------------------------
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
        x = x.squeeze(1)
        scattering_original = x  # shape (batch_size, signal_len, 13)
        # x = x.squeeze().transpose(0, 1)
        # --------------------------------------------------------------------------------------------------------------
        # x = self.encoder(x)
        # x = encoded_x.permute(0, 2, 1)
        # projected_x = self.enc_projection(x)
        # x = projected_x.permute(0, 2, 1)
        # x = projected_x
        # mean = self.linear_mean(x)
        # logvar = self.linear_logvar(x)
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        # z = self.decoder_projection(z)
        # z = z.permute(0, 2, 1)
        # h_ = self.linear_decoder_init(z)
        # h_ = h_.unsqueeze(0).repeat(self.num_LSTM_layers, 1, 1)
        # hidden_decoder = (h_.contiguous(), h_.contiguous())
        x_rec = self.decoder(z)  # shape (64, 300, 13)
        x_rec = x_rec.permute(0, 2, 1)
        # z_temp = z.repeat(1, self.input_size, 1)
        # z_temp = z_temp.view(batch_size, seq_len, self.latent_dim)
        # reconstructed_x = self.decoder(z_temp, hidden_decoder)
        return scattering_original, x_rec, z, mean, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    @staticmethod
    def reconstruction_loss(self, x_input, x_reconstructed):
        return F.mse_loss(x_input, x_reconstructed, reduction='sum')

    @staticmethod
    def kld_loss(mu, log_var):
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())



class VAE_linear(nn.Module):
    def __init__(self, input_seq_size=None, latent_dim=None):
        super(VAE_linear, self).__init__()
        self.input_seq_size = input_seq_size
        self.latent_dim = latent_dim
        self.transform = ScatteringNet(J=11, Q=1, T=(2 ** (11 - 7)), shape=2400)  # todo make this automatic
        self.encoder = ConvLinEncoder(seq_len=self.input_seq_size,
                                      latent_dim=self.latent_dim)

        self.linear_mean = nn.Linear(70, 70)
        self.linear_logvar = nn.Linear(70, 70)

        self.decoder = ConvLinDecoder(latent_dim=70)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def forward(self, x):
        [Sx, Px] = self.transform(x)
        meta = self.transform.meta()
        order0 = np.where(meta['order'] == 0)
        order1 = np.where(meta['order'] == 1)
        order2 = np.where(meta['order'] == 2)
        combined_orders = np.where((meta['order'] == 0) | (meta['order'] == 1))
        selected_orders = torch.from_numpy(order0[0])
        x = Sx[:, :, selected_orders, :]
        x = x.squeeze(1)
        x_scattering = x.squeeze(1)
        x_ = self.encoder(x)
        mean = self.linear_mean(x_)
        logvar = self.linear_logvar(x_)
        z = self.reparameterize(mean, logvar)
        z_ = self.decoder(z)
        z_ = z_.squeeze(1)
        return x_scattering, z_, mean, logvar
