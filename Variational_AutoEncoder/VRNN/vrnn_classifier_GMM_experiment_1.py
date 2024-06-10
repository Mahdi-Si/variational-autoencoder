

import torch.distributions as tdist
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from vrnn_gauss_base import VrnnGaussAbs, VrnnForward
# from torch.jit import fork, wait

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
        self.n_mixtures = 7
        self.logvar_acv = nn.Softplus()
        # feature-extracting transformations (phi_y, phi_u and phi_z)
        self.phi_y = nn.Sequential(
            nn.Linear(self.input_dim, 22),
            nn.LayerNorm(22),
            nn.ELU(),
            nn.Linear(22, 33),
            nn.LayerNorm(33),
            nn.ELU(),
            nn.Linear(33, 44),
            nn.LayerNorm(44),
            nn.ELU(),
            nn.Linear(44, 51),
        )

        self.r_phi_y = nn.Linear(self.input_dim, 51)

        self.phi_h = nn.Sequential(
            nn.Linear(self.h_dim, 90),
            nn.LayerNorm(90),
            nn.ELU(),
            nn.Linear(90, 78),
            nn.LayerNorm(78),
            nn.ELU(),
            nn.Linear(78, 66),
            nn.LayerNorm(66),
            nn.ELU(),
            nn.Linear(66, 51),
        )

        self.r_phi_h = nn.Sequential(
            nn.Linear(self.h_dim, 51)
        )

        self.prior = nn.Sequential(
            nn.Linear(self.h_dim, 79),
            nn.LayerNorm(79),
            nn.ELU(),
            nn.Linear(79, 56),
            nn.LayerNorm(56),
            nn.ELU(),
            nn.Linear(56, 33),
            nn.LayerNorm(33),
            nn.ELU(),
            nn.Linear(33, 9),
        )

        self.r_prior = nn.Sequential(
            nn.Linear(self.h_dim, 9)
        )

        self.prior_mean_logvar = nn.Sequential(
            nn.Linear(9, 2 * 9),
            nn.LayerNorm(2 * 9),
            nn.ELU(),
            nn.Linear(2 * 9, 2 * 9),
            nn.LayerNorm(2 * 9),
            nn.ELU(),
            nn.Linear(2 * 9, 2 * 9),
        )

        # encoder function (phi_enc) -> Inference
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim, 78),
            nn.LayerNorm(78),
            nn.ELU(),
            nn.Linear(78, 51),
            nn.LayerNorm(51),
            nn.ELU(),
            nn.Linear(51, 33),
            nn.LayerNorm(33),
            nn.ELU(),
            nn.Linear(33, 9),
        )

        self.r_enc = nn.Sequential(
            nn.Linear(self.h_dim, 9)
        )

        self.enc_mean_logvar = nn.Sequential(
            nn.Linear(9, 2 * 9),
            nn.LayerNorm(2 * 9),
            nn.ELU(),
            nn.Linear(2 * 9, 2 * 9),
            nn.LayerNorm(2 * 9),
            nn.ELU(),
            nn.Linear(2 * 9, 2 * 9),
        )

        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, 11),
            nn.LayerNorm(11),
            nn.ELU(),
            nn.Linear(11, 33),
            nn.LayerNorm(33),
            nn.ELU(),
            nn.Linear(33, 51),
        )

        self.r_phi_z = nn.Sequential(
            nn.Linear(self.z_dim, 51)
        )

        # decoder function (phi_dec) -> Generation
        self.dec = nn.Sequential(
            nn.Linear(51, 57),
            nn.LayerNorm(57),
            nn.ELU(),
            nn.Linear(57, 63),
            nn.LayerNorm(63),
            nn.ELU(),
            nn.Linear(63, 69),
            nn.LayerNorm(69),
            nn.ELU(),
            nn.Linear(69, 77),
        )

        self.r_dec = nn.Sequential(
            nn.Linear(51, 77)
        )

        self.dec_mean_logvar_pi = nn.Sequential(
            nn.Linear(77, 3 * 77),
            nn.LayerNorm(3 * 77),
            nn.ELU(),
            nn.Linear(3 * 77, 3 * 77),
            nn.LayerNorm(3 * 77),
            nn.ELU(),
            nn.Linear(3 * 77, 3 * 77),
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

        def compute_phi_y_t(y_t_):
            return self.phi_y(y_t_) + self.r_phi_y(y_t_)

        def compute_phi_h_t(h_t_):
            return self.phi_h(h_t_) + self.r_phi_h(h_t_)

        def compute_prior_t(h_t_):
            return self.prior(h_t_) + self.r_prior(h_t_)

        def compute_prior_mean_logvar(prior_t_):
            prior_mean_logvar_ = self.prior_mean_logvar(prior_t_)
            prior_mean_t_ = prior_mean_logvar_[:, :self.z_dim]
            prior_logvar_t_ = self.logvar_acv(prior_mean_logvar_[:, self.z_dim:])
            return prior_mean_t_, prior_logvar_t_

        def compute_enc(phi_y_t_, phi_h_t_):
            enc_t_ = (self.enc(torch.cat([phi_y_t_, phi_h_t_], 1)) +
                      self.r_enc(torch.cat([phi_y_t_, phi_h_t_], 1)))

            enc_mean_logvar_t_ = self.enc_mean_logvar(enc_t_)
            enc_mean_t_ = enc_mean_logvar_t_[:, :self.z_dim]
            enc_logvar_t_ = self.logvar_acv(enc_mean_logvar_t_[:, self.z_dim:])
            return enc_mean_t_, enc_logvar_t_

        # for all time steps
        for t in range(y.size(0)):
            # feature extraction: y_t
            # y original is shape (batch_size, input_size, input_dim)

            y_t = y[t]
            h_t = h[-1]

            phi_y_t_future = torch.jit.fork(compute_phi_y_t, y_t)
            phi_h_t_future = torch.jit.fork(compute_phi_h_t, h_t)
            prior_t_future = torch.jit.fork(compute_prior_t, h_t)

            phi_y_t = torch.jit.wait(phi_y_t_future)
            phi_h_t = torch.jit.wait(phi_h_t_future)
            prior_t = torch.jit.wait(prior_t_future)

            # phi_y_t = self.phi_y(y[t]) + self.r_phi_y(y[t])
            # phi_h_t = self.phi_h(h[-1]) + self.r_phi_h(h[-1])
            # prior_t = self.prior(h[-1]) + self.r_prior(h[-1])

            # prior_mean_logvar = self.prior_mean_logvar(prior_t)
            # prior_mean_t = prior_mean_logvar[:, :self.z_dim]
            # prior_logvar_t = self.logvar_acv(prior_mean_logvar[:, self.z_dim:])
            # # encoder: y_t, h_t -> z_t
            # enc_t = (self.enc(torch.cat([phi_y_t, phi_h_t], 1)) +
            #          self.r_enc(torch.cat([phi_y_t, phi_h_t], 1)))
            #
            # enc_mean_logvar_t = self.enc_mean_logvar(enc_t)
            # enc_mean_t = enc_mean_logvar_t[:, :self.z_dim]
            # enc_logvar_t = self.logvar_acv(enc_mean_logvar_t[:, self.z_dim:])

            prior_mean_logvar_future = torch.jit.fork(compute_prior_mean_logvar, prior_t)
            enc_future = torch.jit.fork(compute_enc, phi_y_t, phi_h_t)

            prior_mean_t, prior_logvar_t = torch.jit.wait(prior_mean_logvar_future)
            enc_mean_t, enc_logvar_t = torch.jit.wait(enc_future)

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


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth_multiplier=1, activation=None):
        super(SeparableConv2D, self).__init__()
        # nn.ELU()
        # filters: 1, depth_multiplier:1, kernel: (1*1)
        self.depthwise = nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size=kernel_size,
                                   groups=in_channels, padding='same')
        self.pointwise = nn.Conv2d(in_channels * depth_multiplier, out_channels, kernel_size=1)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.activation:
            x = self.activation(x)
        return x


# todo think about how you can do custom masking
class CustomMasking(nn.Module):
    def __init__(self, input_dim, out_shape=None):
        super(CustomMasking, self).__init__()
        self.input_dim = input_dim
        self.out_shape = out_shape

    def forward(self, inputs, mask):
        masked_inputs = inputs * mask.unsqueeze(-1).float()
        if self.out_shape is not None:
            return masked_inputs.view(self.out_shape)
        return masked_inputs


class ClassifierBlock(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, conv_kernel_size, conv_depth_multiplier=1,
                 conv_activation=None, lstm_input_dim=None, lstm_h=None,lstm_bidirectional=False):
        super(ClassifierBlock, self).__init__()
        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_depth_multiplier = conv_depth_multiplier
        self.conv_activation = conv_activation
        self.hidden_dims = lstm_h
        self.num_classes = 3  # todo: fix this
        #
        # self.bilstm = nn.LSTM(input_size=None, hidden_size=lstm_h, num_layers=lstm_n_layers, bidirectional=True,
        #                       batch_first=True)
        self.conv_layer = SeparableConv2D(in_channels=self.conv_in_channels,
                                          out_channels=self.conv_out_channels,
                                          kernel_size=self.conv_kernel_size,
                                          depth_multiplier=self.conv_depth_multiplier,
                                          activation=self.conv_activation)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(lstm_input_dim if i == 0 else self.hidden_dims[i - 1], self.hidden_dims[i],
                    bidirectional=lstm_bidirectional, batch_first=True)
            for i in range(len(self.hidden_dims))
        ])

        self.linear = nn.Linear(self.hidden_dims[-1], self.num_classes)
        self.fc_activation = nn.Softmax(dim=2)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.conv_layer(x)  # (batch_size, conv_out_channels, input_size, input_size)
        x = x.squeeze(2).permute(0, 2, 1)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)

        x = self.linear(x)
        return x


class VRNNClassifier(nn.Module):
    def __init__(self, vrnn_model, classifier_model, ):
        super(VRNNClassifier, self).__init__()
        self.vrnn_model = vrnn_model
        self.classifier_model = classifier_model

    def forward(self, x):
        """
        vrnn_model = VRNNGauss(input_dim, input_size, h_dim, z_dim, n_layers, device, log_stat)
        classifier_model = ClassifierBlock(conv_in_channels, conv_out_channels, conv_kernel_size,
                                           conv_depth_multiplier, conv_activation, lstm_input_dim, lstm_h,
                                           lstm_bidirectional)
        model = VRNNClassifier(vrnn_model, classifier_model)
        :param x:
        :return:
        """
        vrnn_output = self.vrnn_model(x)
        z_latent = torch.stack(vrnn_output.z_latent, dim=2)  # shape (batch_size, latent_dim, input_size 150)
        classifier_output = self.classifier_model(z_latent)
        return vrnn_output, classifier_output
