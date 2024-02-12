import torch
from torch import nn
from torch.nn import functional as F
import math
from models.encoder import LSTMEncoder, ConvLinEncoder
from models.decoder import LSTMDecoder, ConvLinDecoder
import numpy as np

from .utils import plot_scattering, \
    EncoderProjection, \
    DecoderProjection, \
    ScatteringNet

class VAE(nn.Module):
    def __init__(self, input_size=None, input_dim=None, dec_hidden_dim=None, enc_hidden_dim=None,
                 latent_size=None, num_LSTM_layers=None):
        super(VAE, self).__init__()
        self.input_size = input_size  # size of the input signal
        self.input_dim = input_dim  # dimension of the input signal
        self.dec_hidden_dim = dec_hidden_dim  # hidden dimension of the decoder LSTM
        self.enc_hidden_dim = enc_hidden_dim
        self.latent_dim = latent_size
        self.num_LSTM_layers = num_LSTM_layers
        self.transform = ScatteringNet(J=11, Q=1, T=(2 ** (11 - 7)), shape=4800)
        self.encoder = LSTMEncoder(input_dim=self.input_dim, hidden_size=self.enc_hidden_dim,
                                   hidden_sizes=[50, 10, 5], num_layers=num_LSTM_layers,
                                   latent_dim=self.latent_dim)
        self.decoder = LSTMDecoder(sequence_length=300, hidden_size=1, hidden_sizes=[10, 50, 100, 1],
                                   num_layers=num_LSTM_layers)
        self.enc_projection = EncoderProjection(seq_len=300, hidden_dim=256, latent_dim=128)
        self.linear_mean = nn.Linear(150, 150)
        self.linear_logvar = nn.Linear(150, 150)
        self.decoder_projection = DecoderProjection(latent_dim=128, seq_len=300)
        self.linear_decoder_init = nn.Linear(self.latent_dim, input_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        fhr = x
        [Sx, Px] = self.transform(x)  # Sx shape(64, 1, 76, 300)
        meta = self.transform.meta()
        order0 = np.where(meta['order'] == 0)
        order1 = np.where(meta['order'] == 1)
        order2 = np.where(meta['order'] == 2)
        combined_orders = np.where((meta['order'] == 0) | (meta['order'] == 1))
        selected_orders = torch.from_numpy(order0[0])
        # x = Sx[0][combined_orders]
        x = Sx[:, :, selected_orders, :]
        x = x.squeeze(1).permute(0, 2, 1)  # (batch_size, 300, 13)
        scattering_original = x
        x = self.encoder(x)
        # x = encoded_x.permute(0, 2, 1)
        # projected_x = self.enc_projection(x)
        # x = projected_x.permute(0, 2, 1)
        # x = projected_x
        mean = self.linear_mean(x)
        logvar = self.linear_logvar(x)
        z = self.reparameterize(mean, logvar)
        # z = self.decoder_projection(z)
        # z = z.permute(0, 2, 1)
        # h_ = self.linear_decoder_init(z)
        # h_ = h_.unsqueeze(0).repeat(self.num_LSTM_layers, 1, 1)
        # hidden_decoder = (h_.contiguous(), h_.contiguous())
        z_ = self.decoder(z)  # shape (64, 300, 13)
        # z_temp = z.repeat(1, self.input_size, 1)
        # z_temp = z_temp.view(batch_size, seq_len, self.latent_dim)
        # reconstructed_x = self.decoder(z_temp, hidden_decoder)
        scattering_original = scattering_original.squeeze(2)
        z_ = z_.squeeze(2)
        return scattering_original, z_, mean, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z


class VAE_linear(nn.Module):
    def __init__(self, input_seq_size=None, latent_dim=None):
        super(VAE_linear, self).__init__()
        self.input_seq_size = input_seq_size
        self.latent_dim = latent_dim
        self.transform = ScatteringNet(J=11, Q=1, T=(2 ** (11 - 7)), shape=2400)  # todo make this automatic
        self.encoder = ConvLinEncoder(seq_len=self.input_seq_size,
                                      latent_dim=self.latent_dim)

        self.linear_mean = nn.Linear(90, 90)
        self.linear_logvar = nn.Linear(90, 90)

        self.decoder = ConvLinDecoder(latent_dim=60)


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
