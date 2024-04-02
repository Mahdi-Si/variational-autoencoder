import torch
import torch.nn as nn
import torch.distributions as tdist
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
from Variational_AutoEncoder.models.dataset_transform import ScatteringTransform


"""implementation of the Variational Recurrent Neural Network (VRNN-Gauss) from https://arxiv.org/abs/1506.02216 using
unimodal isotropic gaussian distributions for inference, prior, and generating models."""


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
    hidden_states: List[torch.Tensor] = None


class VRNN_Gauss(nn.Module):
    def __init__(self, input_dim, input_size, h_dim, z_dim, n_layers, device, log_stat, modify_z=None, modify_h=None,
                 bias=False):
        super(VRNN_Gauss, self).__init__()

        self.input_dim = input_dim
        self.input_size = input_size
        # self.u_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.device = device
        self.modify_z = modify_z
        self.modify_h = modify_h

        self.scattering_transform = ScatteringTransform(input_size=input_size, input_dim=input_dim, log_stat=log_stat,
                                                        device=device)

        # feature-extracting transformations (phi_y, phi_u and phi_z)
        self.phi_y = nn.Sequential(
            nn.Linear(self.input_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),)
        # self.phi_u = nn.Sequential(
        #     nn.Linear(self.u_dim, self.h_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.h_dim, self.h_dim),)
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),)

        # encoder function (phi_enc) -> Inference
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),)
        self.enc_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim))
        self.enc_logvar = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.ReLU(),)

        # prior function (phi_prior) -> Prior
        self.prior = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim))
        self.prior_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim))
        self.prior_logvar = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.ReLU(),)

        # decoder function (phi_dec) -> Generation
        self.dec = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),)
        self.dec_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.input_dim))
        self.dec_logvar = nn.Sequential(
            nn.Linear(self.h_dim, self.input_dim),
            nn.ReLU(),)

        # recurrence function (f_theta) -> Recurrence
        self.rnn = nn.GRU(self.h_dim + self.h_dim, self.h_dim, self.n_layers, bias)  # , batch_first=True)

    def forward(self, y):

        y, meta = self.scattering_transform(y)
        scattering_original = y
        #  batch size
        batch_size = y.shape[0]
        seq_len = y.shape[2]
        # allocation
        loss_ = 0
        loss = torch.zeros(1, device=self.device, requires_grad=True)
        # initialization
        # h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        h = torch.zeros(self.n_layers, y.size(1), self.h_dim, device=self.device)

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_h = []
        all_z_t = []
        all_kld = []
        kld_loss = 0
        nll_loss = 0



        # for all time steps
        for t in range(y.size(0)):
            # feature extraction: y_t
            # phi_y_t = self.phi_y(y[:, :, t])  # y original is shape (batch_size, input_size, input_dim)
            phi_y_t = self.phi_y(y[t])  # should be (input_size, batch_size, input_dim)
            # feature extraction: u_t
            # phi_u_t = self.phi_u(u[:, :, t])

            # encoder: y_t, h_t -> z_t
            enc_t = self.enc(torch.cat([phi_y_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_logvar_t = self.enc_logvar(enc_t)

            # prior: h_t -> z_t (for KLD loss)
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)

            # sampling and reparameterization: get a new z_t
            temp = tdist.Normal(enc_mean_t, enc_logvar_t.exp().sqrt())  # creates a normal distribution object
            z_t = tdist.Normal.rsample(temp)  # sampling from the distribution
            if self.modify_z is not None:
                modify_dims = self.modify_z.get('modify_dims')
                scale = self.modify_z.get('scale')
                shift = self.modify_z.get('shift')
                z_t = self._modify_z(z=z_t, modify_dims=modify_dims, scale=scale, shift=shift)

            # z_t = self._modify_z(z=z_t, modify_dims=[0, 1, 2, 3, 4], scale=0, shift=0)
            # z_t = self._modify_z(z=z_t, modify_dims=[0], scale=10, shift=10)

           # feature extraction: z_t
            phi_z_t = self.phi_z(z_t)

            # decoder: h_t, z_t -> y_t
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_logvar_t = self.dec_logvar(dec_t)
            pred_dist = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt())

            # recurrence: u_t+1, z_t -> h_t+1
            # _, h = self.rnn(torch.cat([phi_u_t, phi_z_t], 1).unsqueeze(0), h)
            _, h = self.rnn(torch.cat([phi_y_t, phi_z_t], 1).unsqueeze(0), h)  # phi_h_t
            # h[:, :, [4, 5, 6, 7, 8]] = 0
            if self.modify_h is not None:
                modify_dims = self.modify_h.get('modify_dims')
                scale = self.modify_h.get('scale')
                shift = self.modify_h.get('shift')
                h = self._modify_h(h=h, modify_dims=modify_dims, scale=scale, shift=shift)

            # computing the loss
            KLD, kld_element = self.kld_gauss(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            # loss_pred = torch.sum(pred_dist.log_prob(y[:, :, t]))
            loss_pred = torch.sum(pred_dist.log_prob(y[t]))
            loss = loss - loss_pred
            kld_loss = kld_loss + KLD

            all_h.append(h)
            all_kld.append(kld_element)
            all_enc_std.append(enc_logvar_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_logvar_t)
            all_z_t.append(z_t)

        results = VrnnForward(
            rec_loss=loss,
            kld_loss=kld_loss,
            nll_loss=nll_loss,
            encoder_mean=all_enc_mean,
            encoder_std=all_enc_std,
            decoder_mean=all_dec_mean,
            decoder_std=all_dec_std,
            kld_values=all_kld,
            Sx=scattering_original,
            Sx_meta=meta,
            z_latent=all_z_t,
            hidden_states=all_h
        )

        return results

    def generate(self, u):
        # get the batch size
        batch_size = u.shape[0]
        # length of the sequence to generate
        seq_len = u.shape[-1]

        # allocation
        sample = torch.zeros(batch_size, self.input_dim, seq_len, device=self.device)
        sample_mu = torch.zeros(batch_size, self.input_dim, seq_len, device=self.device)
        sample_sigma = torch.zeros(batch_size, self.input_dim, seq_len, device=self.device)
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)

        # for all time steps
        for t in range(seq_len):
            # feature extraction: u_t+1
            phi_u_t = self.phi_u(u[:, :, t])

            # prior: h_t -> z_t
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)

            # sampling and reparameterization: get new z_t
            temp = tdist.Normal(prior_mean_t, prior_logvar_t.exp().sqrt())
            z_t = tdist.Normal.rsample(temp)

            # z_t = self._modify_z(z=z_t, modify_dims=[0], scale=10, shift=10)
            # z_t = self._modify_z(z=z_t, modify_dims=[0, 1, 2, 3], scale=10, shift=0)
            # feature extraction: z_t
            phi_z_t = self.phi_z(z_t)

            # decoder: z_t, h_t -> y_t
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_logvar_t = self.dec_logvar(dec_t)
            # store the samples
            temp = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt())
            sample[:, :, t] = tdist.Normal.rsample(temp)
            # store mean and std
            sample_mu[:, :, t] = dec_mean_t
            sample_sigma[:, :, t] = dec_logvar_t.exp().sqrt()

            # recurrence: u_t+1, z_t -> h_t+1
            _, h = self.rnn(torch.cat([phi_u_t, phi_z_t], 1).unsqueeze(0), h)

        return sample, sample_mu, sample_sigma

    @staticmethod
    def kld_gauss(mu_q, logvar_q, mu_p, logvar_p):
        # Goal: Minimize KL divergence between q_pi(z|xi) || p(z|xi)
        # This is equivalent to maximizing the ELBO: - D_KL(q_phi(z|xi) || p(z)) + Reconstruction term
        # This is equivalent to minimizing D_KL(q_phi(z|xi) || p(z))
        term1 = logvar_p - logvar_q - 1
        term2 = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        final_term = term1 + term2
        kld = 0.5 * torch.sum(final_term)

        return kld, final_term

    def init_rnn_output(self, batch_size, seq_len):
        phi_h_t = torch.zeros(batch_size, seq_len, self.h_dim).to(self.device)

        return phi_h_t

    @staticmethod
    def _modify_z(z, modify_dims, shift, scale):
        for i in modify_dims:
            z[:, i] = scale[i] * z[:, i] + shift[i]
        return z

    @staticmethod
    def _modify_h(h, modify_dims, shift, scale):
        if h.dim == 3:
            for i in modify_dims:
                h[:, :, i] = scale * h[:, :, i] + shift
        elif h.dim == 2:
            for i in modify_dims:
                h[:, i] = scale * h[:, i] + shift
        return h
