

import torch.distributions as tdist
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from vrnn_gauss_base import VrnnGaussAbs, VrnnForward

"""implementation of the Variational Recurrent Neural Network (VRNN-Gauss) from https://arxiv.org/abs/1506.02216 using
uni-modal isotropic gaussian distributions for inference, prior, and generating models."""


class VRNNGauss(VrnnGaussAbs):
    def __init__(self, input_dim, input_size, h_dim, z_dim, n_layers, device, log_stat, modify_z=None, modify_h=None,
                 bias=False):
        super(VRNNGauss, self).__init__(input_dim=input_dim,
                                        input_size=input_size,
                                        h_dim=h_dim,
                                        z_dim=z_dim,
                                        n_layers=n_layers,
                                        device=device,
                                        log_stat=log_stat,
                                        modify_z=None,
                                        modify_h=None,
                                        bias=False)
        self.n_mixtures = 5
        self.logvar_acv = nn.Softplus()
        # feature-extracting transformations (phi_y, phi_u and phi_z)
        self.phi_y = nn.Sequential(
            nn.Linear(self.input_dim, 14),
            nn.LayerNorm(14),
            nn.ReLU(),
            nn.Linear(14, 17),
            nn.LayerNorm(17),
            nn.ReLU(),
            nn.Linear(17, 18)

        )

        self.r_phi_y = nn.Linear(self.input_dim, 18)

        self.phi_h = nn.Sequential(
            nn.Linear(self.h_dim, 30),
            nn.LayerNorm(30),
            nn.ReLU(),
            nn.Linear(30, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Linear(24, 18),
        )

        self.r_phi_h = nn.Sequential(
            nn.Linear(self.h_dim, 18)
        )

        self.prior = nn.Sequential(
            nn.Linear(self.h_dim, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Linear(24, 18),
            nn.LayerNorm(18),
            nn.ReLU(),
            nn.Linear(18, 11),
            nn.LayerNorm(11),
        )

        self.r_prior = nn.Sequential(
            nn.Linear(self.h_dim, 11)
        )

        self.prior_mean_logvar = nn.Sequential(
            nn.Linear(11, 2 * 9),
            nn.LayerNorm(2 * 9),
            nn.ReLU(),
            nn.Linear(2 * 9, 2 * 6),
            nn.LayerNorm(2 * 6),
            nn.ReLU(),
            nn.Linear(2 * 6, 2 * 3),
        )

        # encoder function (phi_enc) -> Inference
        self.enc = nn.Sequential(
            nn.Linear(36, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Linear(24, 18),
            nn.LayerNorm(18),
            nn.ReLU(),
            nn.Linear(18, 11),
        )

        self.r_enc = nn.Sequential(
            nn.Linear(self.h_dim, 11)
        )

        self.enc_mean_logvar = nn.Sequential(
            nn.Linear(11, 2 * 9),
            nn.LayerNorm(2 * 9),
            nn.ReLU(),
            nn.Linear(2 * 9, 2 * 6),
            nn.LayerNorm(2 * 6),
            nn.ReLU(),
            nn.Linear(2 * 6, 2 * 3),
        )

        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, 6),
            nn.LayerNorm(6),
            nn.ReLU(),
            nn.Linear(6, 9),
            nn.LayerNorm(9),
            nn.ReLU(),
            nn.Linear(9, 11),
            nn.LayerNorm(11),
            nn.ReLU(),
            nn.Linear(11, 18),
        )

        self.r_phi_z = nn.Sequential(
            nn.Linear(self.z_dim, 18)
        )

        # decoder function (phi_dec) -> Generation
        self.dec = nn.Sequential(
            nn.Linear(18, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Linear(24, 30),
            nn.LayerNorm(30),
            nn.ReLU(),
            nn.Linear(30, 36),
        )

        self.r_dec = nn.Sequential(
            nn.Linear(18, 36)
        )

        self.dec_mean_logvar_pi = nn.Sequential(
            nn.Linear(36, 3 * 44),
            nn.LayerNorm(3 * 44),
            nn.ReLU(),
            nn.Linear(3 * 44, 3 * 48),
            nn.LayerNorm(3 * 48),
            nn.ReLU(),
            nn.Linear(3 * 48, 3 * 55),
        )
        self.dec_pi_activation = nn.Softmax(dim=2)

        self.rnn = nn.LSTM(self.h_dim, self.h_dim, self.n_layers, bias)  # , batch_first=True

    def forward(self, y):
        y, meta = self.scattering_transform(y)
        batch_size = y.shape[1]
        scattering_original = y
        loss = torch.zeros(1, device=self.device, requires_grad=True)
        h = torch.randn(self.n_layers, y.size(1), self.h_dim, device=self.device)
        c = torch.randn(self.n_layers, y.size(1), self.h_dim, device=self.device)

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_h = []
        all_z_t = []
        all_kld = []
        kld_loss = 0

        # for all time steps
        for t in range(y.size(0)):
            # feature extraction: y_t
            # y original is shape (batch_size, input_size, input_dim)
            phi_y_t = self.phi_y(y[t]) + self.r_phi_y(y[t])

            phi_h_t = self.phi_h(h[-1]) + self.r_phi_h(h[-1])

            prior_t = self.prior(h[-1]) + self.r_prior(h[-1])
            prior_mean_logvar = self.prior_mean_logvar(prior_t)
            prior_mean_t = prior_mean_logvar[:, :self.z_dim]
            prior_logvar_t = self.logvar_acv(prior_mean_logvar[:, self.z_dim:])
            # encoder: y_t, h_t -> z_t
            enc_t = (self.enc(torch.cat([phi_y_t, phi_h_t], 1)) +
                     self.r_enc(torch.cat([phi_y_t, phi_h_t], 1)))

            enc_mean_logvar_t = self.enc_mean_logvar(enc_t)
            enc_mean_t = enc_mean_logvar_t[:, :self.z_dim]
            enc_logvar_t = self.logvar_acv(enc_mean_logvar_t[:, self.z_dim:])

            # sampling and reparameterization: get a new z_t
            temp = tdist.Normal(enc_mean_t, enc_logvar_t.exp().sqrt())  # creates a normal distribution object
            z_t = tdist.Normal.rsample(temp)  # sampling from the distribution
            if self.modify_z is not None:
                modify_dims = self.modify_z.get('modify_dims')
                scale = self.modify_z.get('scale')
                shift = self.modify_z.get('shift')
                z_t = self._modify_z(z=z_t, modify_dims=modify_dims, scale=scale, shift=shift)

            # feature extraction: z_t
            phi_z_t = self.phi_z(z_t) + self.r_phi_z(z_t)

            dec_t = self.dec(phi_z_t) + self.r_dec(phi_z_t)
            dec_mean_logvar_pi_t = self.dec_mean_logvar_pi(dec_t)

            dec_mean_t_gmm = (
                dec_mean_logvar_pi_t[:, :self.input_dim * self.n_mixtures]
            ).view(batch_size, self.input_dim, self.n_mixtures)

            dec_logvar_t_gmm = (
                self.logvar_acv(
                    dec_mean_logvar_pi_t[:, self.input_dim * self.n_mixtures:2 * self.input_dim * self.n_mixtures]
                )
            ).view(batch_size, self.input_dim, self.n_mixtures)

            dec_pi_t_gmm = (
                dec_mean_logvar_pi_t[:, 2 * self.input_dim * self.n_mixtures:3 * self.input_dim * self.n_mixtures]
            ).view(batch_size, self.input_dim, self.n_mixtures)

            dec_pi_t = self.dec_pi_activation(dec_pi_t_gmm)
            # recurrence: y_t, z_t -> h_t+1
            # _, h = self.rnn(torch.cat([phi_y_t, phi_z_t], 1).unsqueeze(0), h)  # phi_h_t
            _, (h, c) = self.rnn(torch.cat([phi_y_t, phi_z_t], 1).unsqueeze(0), (h, c))

            if self.modify_h is not None:
                modify_dims = self.modify_h.get('modify_dims')
                scale = self.modify_h.get('scale')
                shift = self.modify_h.get('shift')
                h = self._modify_h(h=h, modify_dims=modify_dims, scale=scale, shift=shift)

            dec_pi_t = self.dec_pi_activation(dec_pi_t_gmm)
            # computing the loss
            KLD, kld_element = self.kld_gauss_(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            loss_pred = self.loglikelihood_gmm(y[t], dec_mean_t_gmm, dec_logvar_t_gmm, dec_pi_t)
            loss = loss - loss_pred
            kld_loss = kld_loss + KLD

            # GMM  ====================================================================
            dec_mean_t = torch.sum(dec_pi_t * dec_mean_t_gmm, dim=-1)
            dec_logvar_t = dec_mean_t
            # end   ====================================================================

            all_h.append(h)
            all_kld.append(kld_element)
            all_enc_std.append(enc_logvar_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_logvar_t)
            all_z_t.append(z_t)
        results = VrnnForward(
            rec_loss=loss,  # (1,)
            kld_loss=kld_loss,  # ()
            nll_loss=loss,
            encoder_mean=all_enc_mean,  # list(input_size) -> each element: (batch_size, latent_dim)
            encoder_std=all_enc_std,  # list(input_size) -> each element: (batch_size, latent_dim)
            decoder_mean=all_dec_mean,  # list(input_size) -> each element: (batch_size, input_dim)
            decoder_std=all_dec_std,  # list(input_size) -> each element: (batch_size, input_dim)
            kld_values=all_kld,  # list(input_size) -> each element: (batch_size, latent_dim)
            Sx=scattering_original,  # (input_size, batch_size, input_dim)
            Sx_meta=meta,
            z_latent=all_z_t,  # list(input_size) -> each element: (batch_size, latent_dim)
            hidden_states=all_h   # list(input_size) -> each element: (n_layers, batch_size, input_dim)
        )

        return results

    def generate(self, input_size, batch_size):
        # sample = torch.zeros(batch_size, self.input_dim, input_size, device=self.device)
        # sample_mu = torch.zeros(batch_size, self.input_dim, input_size, device=self.device)
        # sample_sigma = torch.zeros(batch_size, self.input_dim, input_size, device=self.device)
        sample = torch.zeros(input_size, batch_size, self.input_dim, device=self.device)
        sample_mu = torch.zeros(input_size, batch_size, self.input_dim, device=self.device)
        sample_sigma = torch.zeros(input_size, batch_size, self.input_dim, device=self.device)

        h = torch.randn(self.n_layers, batch_size, self.h_dim, device=self.device)
        c = torch.randn(self.n_layers, batch_size, self.h_dim, device=self.device)

        # prior_mean_t = torch.zeros([batch_size, self.z_dim], device=self.device)
        # prior_logvar_t = torch.zeros([batch_size, self.z_dim], device=self.device)

        # for all time steps
        for t in range(input_size):
            prior_t = self.prior(h[-1]) + self.r_prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)
            # sampling and reparameterization: get new z_t
            temp = tdist.Normal(prior_mean_t, prior_logvar_t.exp().sqrt())
            z_t = tdist.Normal.rsample(temp)

            # feature extraction: z_t
            phi_z_t = self.phi_z(z_t) + self.r_phi_z(z_t)

            dec_t = self.dec(phi_z_t) + self.r_dec(phi_z_t)
            dec_mean_t = self.dec_mean(dec_t)
            dec_logvar_t = self.dec_logvar(dec_t)
            temp = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt())

            phi_y_t = self.phi_y(dec_mean_t) + self.r_phi_y(dec_mean_t)

            sample[t] = tdist.Normal.rsample(temp)
            sample_mu[t] = dec_mean_t
            sample_sigma[t] = dec_logvar_t
            _, (h, c) = self.rnn(torch.cat([phi_y_t, phi_z_t], 1).unsqueeze(0), (h, c))

        return sample, sample_mu, sample_sigma

    def loglikelihood_gmm(self, x, mu, logvar, pi):
        loglike = 0
        # for all data channels
        for n in range(x.shape[1]):
            # likelihood of a single mixture at evaluation point
            pred_dist = tdist.Normal(mu[:, n, :], logvar[:, n, :].exp().sqrt())
            x_mod = torch.mm(x[:, n].unsqueeze(1), torch.ones(1, self.n_mixtures, device=self.device))
            like = pred_dist.log_prob(x_mod)
            # weighting by probability of mixture and summing
            temp = (pi[:, n, :] * like)
            temp = temp.sum()
            # log-likelihood added to previous log-likelihoods
            loglike = loglike + temp

        return loglike

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

    # def reconstruct(self, dec_mean, dec_logvar, dec_pi):
    #     """
    #     Reconstruct the input signal from the decoder outputs.
    #
    #     Parameters:
    #     dec_mean (torch.Tensor): Mean values for the GMM components (batch_size, input_dim, n_mixtures)
    #     dec_logvar (torch.Tensor): Log variance values for the GMM components (batch_size, input_dim, n_mixtures)
    #     dec_pi (torch.Tensor): Mixture coefficients for the GMM components (batch_size, input_dim, n_mixtures)
    #
    #     Returns:
    #     torch.Tensor: Reconstructed signal (batch_size, input_dim)
    #     """
    #     samples = []
    #     for n in range(dec_mean.shape[1]):
    #         # likelihood of a single mixture at evaluation point
    #         pred_dist = tdist.Normal(dec_mean[:, n, :], dec_logvar[:, n, :].exp().sqrt())
    #         x_mod = torch.mm(x[:, n].unsqueeze(1), torch.ones(1, self.n_mixtures, device=self.device))
    #         # like = pred_dist.log_prob(x_mod)
    #         # weighting by probability of mixture and summing
    #         # temp = (dec_pi[:, n, :] * like)
    #         # temp = temp.sum()
    #         # log-likelihood added to previous log-likelihoods
    #         loglike = loglike + temp
    #
    #
    #     # Convert log variances to standard deviations
    #     dec_std = torch.exp(0.5 * dec_logvar)
    #
    #     # Sample from each mixture component
    #     samples = []
    #     for k in range(self.n_mixtures):
    #         dist = tdist.Normal(dec_mean[:, :, k], dec_std[:, :, k])
    #         sample = dist.rsample()
    #         samples.append(sample)
    #
    #     # Stack the samples along a new dimension
    #     samples = torch.stack(samples, dim=-1)
    #
    #     # Weight the samples by the mixture coefficients and sum them
    #     reconstructed_signal = torch.sum(dec_pi * samples, dim=-1)
    #
    #     return reconstructed_signal