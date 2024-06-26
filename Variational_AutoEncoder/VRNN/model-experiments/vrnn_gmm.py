import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as tdist
import torch
import math
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Variational_AutoEncoder.models.misc import ScatteringNet
from Variational_AutoEncoder.utils.data_utils import plot_scattering
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from Variational_AutoEncoder.models.dataset_transform import ScatteringTransform


"""implementation of the Variational Recurrent Neural Network (VRNN-GMM) from https://arxiv.org/abs/1506.02216 using
Gaussian mixture distributions with fixed number of mixtures for inference, prior, and generating models."""


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


class VRNN_GMM(nn.Module):
    def __init__(self,  input_dim, input_size, h_dim, z_dim, n_layers, device, log_stat, bias=False):
        super(VRNN_GMM, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.device = device
        self.n_mixtures = 5

        self.scattering_transform = ScatteringTransform(input_size=input_size, input_dim=input_dim, log_stat=log_stat,
                                                        device=device)

        # feature-extracting transformations (phi_y, phi_u and phi_z)
        self.phi_y = nn.Sequential(
            nn.Linear(self.input_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim))
        # self.phi_u = nn.Sequential(
        #     nn.Linear(self.u_dim, self.h_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.h_dim, self.h_dim))
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim))

        # encoder function (phi_enc) -> Inference
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(), )
        self.enc_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim))
        self.enc_logvar = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.ReLU(), )

        # prior function (phi_prior) -> Prior
        self.prior = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(), )
        self.prior_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim))
        self.prior_logvar = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.ReLU(), )

        # decoder function (phi_dec) -> Generation
        self.dec = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(), )
        self.dec_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.input_dim * self.n_mixtures), )
        self.dec_logvar = nn.Sequential(
            nn.Linear(self.h_dim, self.input_dim * self.n_mixtures),
            nn.ReLU(), )
        self.dec_pi = nn.Sequential(
            nn.Linear(self.h_dim, self.input_dim * self.n_mixtures),
            nn.Softmax(dim=1)
        )

        # recurrence function (f_theta) -> Recurrence
        self.rnn = nn.GRU(self.h_dim + self.h_dim, self.h_dim, self.n_layers, bias)

    def forward(self, y):

        y, meta = self.scattering_transform(y)
        scattering_original = y
        # batch_size = y.size(0)
        # seq_len = y.shape[-1]

        # allocation
        loss_ = 0
        loss = torch.zeros(1, device=self.device, requires_grad=True)
        # initialization
        h = torch.zeros(self.n_layers, y.size(1), self.h_dim, device=self.device)

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_h = []
        all_z_t = []
        all_kld = []
        all_dec_reconstructed = []
        kld_loss = 0
        nll_loss = 0

        # for all time steps
        for t in range(y.size(0)):
            # feature extraction: y_t
            phi_y_t = self.phi_y(y[t])
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
            temp = tdist.Normal(enc_mean_t, enc_logvar_t.exp().sqrt())
            z_t = tdist.Normal.rsample(temp)
            # feature extraction: z_t
            phi_z_t = self.phi_z(z_t)
            # z_t = self._modify_z(z=z_t, modify_dims=[0, 1, 2, 3, 4], scale=0, shift=0)
            # z_t = self._modify_z(z=z_t, modify_dims=[0], scale=10, shift=10)

            # decoder: h_t, z_t -> y_t
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))

            # todo it was batch_size, you change to y.size(1), check the view option
            # dec_mean_t = self.dec_mean(dec_t).view(y.size(1), self.input_dim, self.n_mixtures)
            # dec_logvar_t = self.dec_logvar(dec_t).view(y.size(1), self.input_dim, self.n_mixtures)
            # dec_pi_t = self.dec_pi(dec_t).view(y.size(1), self.input_dim, self.n_mixtures)

            dec_mean_t = self.dec_mean(dec_t).view(self.input_dim, y.size(1), self.n_mixtures)
            dec_logvar_t = self.dec_logvar(dec_t).view(self.input_dim, y.size(1), self.n_mixtures)
            dec_pi_t = self.dec_pi(dec_t).view(self.input_dim, y.size(1), self.n_mixtures)
            reconstructed_t = torch.sum(dec_mean_t * dec_pi_t, dim=2)
            # recurrence: u_t+1, z_t -> h_t+1
            _, h = self.rnn(torch.cat([phi_y_t, phi_z_t], 1).unsqueeze(0), h)

            # computing the loss
            KLD, kld_element = self.kld_gauss(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            loss_pred = self.loglikelihood_gmm(y[t], dec_mean_t, dec_logvar_t, dec_pi_t)
            # loss += - loss_pred
            loss = loss - loss_pred
            kld_loss = kld_loss + KLD
            all_h.append(h)
            all_kld.append(kld_element)
            all_enc_std.append(enc_logvar_t)
            all_enc_mean.append(enc_mean_t)
            # all_dec_mean.append(dec_mean_t)
            # all_dec_std.append(dec_logvar_t)
            all_dec_mean.append(dec_t)
            all_dec_std.append(dec_t)
            all_z_t.append(z_t)
            all_dec_reconstructed.append(reconstructed_t.permute(1, 0))
        results = VrnnForward(
            rec_loss=loss,  # (1, )
            kld_loss=kld_loss,  # ()
            nll_loss=nll_loss,
            encoder_mean=all_enc_mean,  # list(input_size) -> each element: (batch_size, latent_dim)
            encoder_std=all_enc_std,  # list(input_size) -> each element: (batch_size, latent_dim)
            decoder_mean=all_dec_reconstructed,  # list(150) -> each element: (input_dim, batch_size, n_mixtures)
            decoder_std=all_dec_std,  # list(150) -> each element: (input_dim, batch_size, n_mixtures)
            kld_values=all_kld,  # list(input_size) -> each element: (batch_size, latent_dim)
            Sx=scattering_original,  # (input_size, batch_size, input_dim)
            Sx_meta=meta,
            z_latent=all_z_t,  # list(input_size) -> each element: (batch_size, latent_dim)
            hidden_states=all_h  # list(input_size) -> each element: (n_layers, batch_size, input_dim)
        )

        return results

    def generate(self, u):
        # get the batch size
        batch_size = u.shape[0]
        # length of the sequence to generate
        seq_len = u.shape[-1]

        # allocation
        sample = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        sample_mu = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        sample_sigma = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)

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
            # feature extraction: z_t
            phi_z_t = self.phi_z(z_t)

            # decoder: z_t, h_t -> y_t
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t).view(batch_size, self.y_dim, self.n_mixtures)
            dec_logvar_t = self.dec_logvar(dec_t).view(batch_size, self.y_dim, self.n_mixtures)
            dec_pi_t = self.dec_pi(dec_t).view(batch_size, self.y_dim, self.n_mixtures)

            # store the samples
            sample[:, :, t], sample_mu[:, :, t], sample_sigma[:, :, t] = self._reparameterized_sample_gmm(dec_mean_t,
                                                                                                          dec_logvar_t,
                                                                                                          dec_pi_t)

            # recurrence: u_t+1, z_t -> h_t+1
            _, h = self.rnn(torch.cat([phi_u_t, phi_z_t], 1).unsqueeze(0), h)

        return sample, sample_mu, sample_sigma

    def _reparameterized_sample_gmm(self, mu, logvar, pi):

        # select the mixture indices
        alpha = torch.distributions.Categorical(pi).sample()

        # select the mixture indices
        idx = logvar.shape[-1]
        raveled_index = torch.arange(len(alpha.flatten()), device=self.device) * idx + alpha.flatten()
        logvar_sel = logvar.flatten()[raveled_index]
        mu_sel = mu.flatten()[raveled_index]

        # get correct dimensions
        logvar_sel = logvar_sel.view(logvar.shape[:-1])
        mu_sel = mu_sel.view(mu.shape[:-1])

        # resample
        temp = tdist.Normal(mu_sel, logvar_sel.exp().sqrt())
        sample = tdist.Normal.rsample(temp)

        return sample, mu_sel, logvar_sel.exp().sqrt()

    def loglikelihood_gmm(self, x, mu, logvar, pi):
        # init
        loglike = 0
        # x (batch_size, input_dim)
        # logvar (input_dim, batch_size, n_gmm)
        # mu (input_dim, batch_size, n_gmm)
        # pi (input_dim, batch_size, n_gmm)
        # for all data channels
        list_rec = []
        for n in range(x.shape[1]):
            # likelihood of a single mixture at evaluation point
            # pred_dist = tdist.Normal(mu[:, n, :], logvar[:, n, :].exp().sqrt())
            pred_dist = tdist.Normal(mu[n, :, :], logvar[n, :, :].exp().sqrt())
            x_mod = torch.mm(x[:, n].unsqueeze(1), torch.ones(1, self.n_mixtures, device=self.device))
            like = pred_dist.log_prob(x_mod)
            # weighting by probability of mixture and summing
            temp = (pi[n, :, :] * like)
            list_rec.append(temp)
            temp = temp.sum()
            # log-likelihood added to previous log-likelihoods
            loglike = loglike + temp

        return loglike

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

    @staticmethod
    def _modify_z(z, modify_dims, shift, scale):
        for i in modify_dims:
            z[:, i] = scale * z[:, i] + shift
        return z
