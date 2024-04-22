

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

        # feature-extracting transformations (phi_y, phi_u and phi_z)
        self.phi_y = nn.Sequential(
            nn.Linear(self.input_dim, int(self.h_dim / 3)),
            nn.ReLU(),
            nn.Linear(int(self.h_dim / 3), int(self.h_dim / 2)),
        )
        # self.phi_u = nn.Sequential(
        #     nn.Linear(self.u_dim, self.h_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.h_dim, self.h_dim),)
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, int(self.h_dim / 3)),
            nn.ReLU(),
            nn.Linear(int(self.h_dim / 3), int(self.h_dim / 2)),
            # nn.ReLU(),
            # nn.Linear(int(self.h_dim / 2), int(2 * self.h_dim / 3))
        )

        self.phi_h = nn.Sequential(
            nn.Linear(self.h_dim, int(2 * self.h_dim / 3)),
            nn.ReLU(),
            nn.Linear(int(2 * self.h_dim / 3), int(self.h_dim / 2)),
        )

        # encoder function (phi_enc) -> Inference
        self.enc = nn.Sequential(
            nn.Linear(2 * int(self.h_dim / 2), int(2 * self.h_dim / 3)),
            nn.ReLU(),
            nn.Linear(int(2 * self.h_dim / 3), int(self.h_dim / 2)),
            nn.ReLU(),
        )
        self.enc_mean = nn.Sequential(
            nn.Linear(int(self.h_dim / 2), int(self.h_dim / 3)),
            nn.ReLU(),
            nn.Linear(int(self.h_dim / 3), self.z_dim)
        )
        self.enc_logvar = nn.Sequential(
            nn.Linear(int(self.h_dim / 2), int(self.h_dim / 3)),
            nn.ReLU(),
            nn.Linear(int(self.h_dim / 3), self.z_dim),
            # nn.ReLU(),
            nn.Softplus(),
        )

        # decoder function (phi_dec) -> Generation
        self.dec = nn.Sequential(
            nn.Linear(int(self.h_dim / 2), int(2 * self.h_dim / 3)),
            nn.ReLU(),
            nn.Linear(int(2 * self.h_dim / 3), self.h_dim),
            nn.ReLU(),)
        self.dec_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.input_dim))
        self.dec_logvar = nn.Sequential(
            nn.Linear(self.h_dim, self.input_dim),
            # nn.ReLU(),
            nn.Softplus(),
        )

        # recurrence function (f_theta) -> Recurrence
        # self.rnn = nn.GRU(self.h_dim + self.h_dim, self.h_dim, self.n_layers, bias)  # , batch_first=True
        self.rnn = nn.LSTM(self.h_dim, self.h_dim, self.n_layers, bias)  # , batch_first=True

    def forward(self, y):
        y, meta = self.scattering_transform(y)
        scattering_original = y
        loss = torch.zeros(1, device=self.device, requires_grad=True)
        # initialization
        # h = torch.zeros(self.n_layers, y.size(1), self.h_dim, device=self.device)

        # h = torch.zeros(self.n_layers, y.size(1), self.h_dim, device=self.device)
        # c = torch.zeros(self.n_layers, y.size(1), self.h_dim, device=self.device)
        h = torch.randn(self.n_layers, y.size(1), self.h_dim, device=self.device)
        c = torch.randn(self.n_layers, y.size(1), self.h_dim, device=self.device)

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_h = []
        all_z_t = []
        all_kld = []
        kld_loss = 0

        # prior_mean_t = torch.zeros([y.size(1), self.z_dim], device=self.device)
        # prior_logvar_t = torch.zeros([y.size(1), self.z_dim], device=self.device)

        prior_mean_t = torch.randn([y.size(1), self.z_dim], device=self.device) * 0.01
        prior_logvar_t = torch.full([y.size(1), self.z_dim], -1.0, device=self.device)  # log(var) = -1 => var = exp(-1)

        # for all time steps
        for t in range(y.size(0)):
            # feature extraction: y_t
            # phi_y_t = self.phi_y(y[:, :, t])  # y original is shape (batch_size, input_size, input_dim)
            phi_y_t = self.phi_y(y[t])  # should be (input_size, batch_size, input_dim)
            # feature extraction: u_t
            # phi_u_t = self.phi_u(u[:, :, t])
            phi_h_t = self.phi_h(h[-1])
            # encoder: y_t, h_t -> z_t
            enc_t = self.enc(torch.cat([phi_y_t, phi_h_t], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_logvar_t = self.enc_logvar(enc_t)

            # sampling and reparameterization: get a new z_t
            temp = tdist.Normal(enc_mean_t, enc_logvar_t.exp().sqrt())  # creates a normal distribution object
            z_t = tdist.Normal.rsample(temp)  # sampling from the distribution
            if self.modify_z is not None:
                modify_dims = self.modify_z.get('modify_dims')
                scale = self.modify_z.get('scale')
                shift = self.modify_z.get('shift')
                z_t = self._modify_z(z=z_t, modify_dims=modify_dims, scale=scale, shift=shift)

            # feature extraction: z_t
            phi_z_t = self.phi_z(z_t)

            # decoder: h_t, z_t -> y_t
            # dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_t = self.dec(phi_z_t)
            dec_mean_t = self.dec_mean(dec_t)
            dec_logvar_t = self.dec_logvar(dec_t)
            pred_dist = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt())

            # recurrence: y_t, z_t -> h_t+1
            # _, h = self.rnn(torch.cat([phi_y_t, phi_z_t], 1).unsqueeze(0), h)  # phi_h_t
            _, (h, c) = self.rnn(torch.cat([phi_y_t, phi_z_t], 1).unsqueeze(0), (h, c))

            if self.modify_h is not None:
                modify_dims = self.modify_h.get('modify_dims')
                scale = self.modify_h.get('scale')
                shift = self.modify_h.get('shift')
                h = self._modify_h(h=h, modify_dims=modify_dims, scale=scale, shift=shift)

            # computing the loss
            KLD, kld_element = self.kld_gauss_(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
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
