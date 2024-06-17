

import torch.distributions as tdist
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
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
        self.logvar_acv = nn.Softplus()
        self.phi_y = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 21),
            nn.LayerNorm(21),
            nn.ReLU(),
            nn.Linear(21, 26),
            nn.LayerNorm(26),
            nn.ReLU(),
            nn.Linear(26, 30),
        )

        self.r_phi_y = nn.Linear(self.input_dim, 30)

        self.phi_h = nn.Sequential(
            nn.Linear(self.h_dim, 54),
            nn.LayerNorm(54),
            nn.ReLU(),
            nn.Linear(54, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Linear(48, 42),
            nn.LayerNorm(42),
            nn.ReLU(),
            nn.Linear(42, 36),
            nn.LayerNorm(36),
            nn.ReLU(),
            nn.Linear(36, 30),
        )

        self.r_phi_h = nn.Sequential(
            nn.Linear(self.h_dim, 30)
        )

        self.prior = nn.Sequential(
            nn.Linear(self.h_dim, 54),
            nn.LayerNorm(54),
            nn.ReLU(),
            nn.Linear(54, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Linear(48, 42),
            nn.LayerNorm(42),
            nn.ReLU(),
            nn.Linear(42, 36),
            nn.LayerNorm(36),
            nn.ReLU(),
            nn.Linear(36, 30),
        )

        self.r_prior = nn.Sequential(
            nn.Linear(self.h_dim, 30)
        )

        self.prior_mean = nn.Sequential(
            nn.Linear(30, 25),
            nn.LayerNorm(25),
            nn.ReLU(),
            nn.Linear(25, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.LayerNorm(15),
            nn.ReLU(),
            nn.Linear(15, 10),
            nn.LayerNorm(10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

        self.r_prior_mean = nn.Sequential(
            nn.Linear(30, 5)
        )

        self.prior_logvar = nn.Sequential(
            nn.Linear(30, 25),
            nn.LayerNorm(25),
            nn.ReLU(),
            nn.Linear(25, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.LayerNorm(15),
            nn.ReLU(),
            nn.Linear(15, 10),
            nn.LayerNorm(10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

        self.r_prior_logvar = nn.Sequential(
            nn.Linear(30, 5)
        )

        self.enc = nn.Sequential(
            nn.Linear(self.h_dim, 54),
            nn.LayerNorm(54),
            nn.ReLU(),
            nn.Linear(54, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Linear(48, 42),
            nn.LayerNorm(42),
            nn.ReLU(),
            nn.Linear(42, 36),
            nn.LayerNorm(36),
            nn.ReLU(),
            nn.Linear(36, 30),
        )

        self.r_enc = nn.Sequential(
            nn.Linear(self.h_dim, 30)
        )

        # ================================================================
        self.enc_mean = nn.Sequential(
            nn.Linear(30, 25),
            nn.LayerNorm(25),
            nn.ReLU(),
            nn.Linear(25, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.LayerNorm(15),
            nn.ReLU(),
            nn.Linear(15, 10),
            nn.LayerNorm(10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

        self.r_enc_mean = nn.Sequential(
            nn.Linear(30, 5)
        )

        self.enc_logvar = nn.Sequential(
            nn.Linear(30, 25),
            nn.LayerNorm(25),
            nn.ReLU(),
            nn.Linear(25, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.LayerNorm(15),
            nn.ReLU(),
            nn.Linear(15, 10),
            nn.LayerNorm(10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

        self.r_enc_logvar = nn.Sequential(
            nn.Linear(30, 5)
        )

        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, 10),
            nn.LayerNorm(10),
            nn.ReLU(),
            nn.Linear(10, 15),
            nn.LayerNorm(15),
            nn.ReLU(),
            nn.Linear(15, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Linear(20, 25),
            nn.LayerNorm(25),
            nn.ReLU(),
            nn.Linear(25, 30),
        )

        self.r_phi_z = nn.Sequential(
            nn.Linear(self.z_dim, 30)
        )

        # decoder function (phi_dec) -> Generation
        self.dec = nn.Sequential(
            nn.Linear(30, 27),
            nn.LayerNorm(27),
            nn.ELU(),
            nn.Linear(27, 24),
            nn.LayerNorm(24),
            nn.ELU(),
            nn.Linear(24, 21),
            nn.LayerNorm(21),
            nn.ELU(),
            nn.Linear(21, 19),
        )

        self.r_dec = nn.Sequential(
            nn.Linear(30, 19)
        )

        self.dec_mean = nn.Sequential(
            nn.Linear(19, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 13),
            nn.LayerNorm(13),
            nn.ReLU(),
            nn.Linear(13, 11),
        )

        self.r_dec_mean = nn.Sequential(
            nn.Linear(19, 11),
        )

        self.dec_logvar = nn.Sequential(
            nn.Linear(19, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 13),
            nn.LayerNorm(13),
            nn.ReLU(),
            nn.Linear(13, 11),
        )

        self.r_dec_logvar = nn.Sequential(
            nn.Linear(19, 11),
        )

        self.rnn = nn.LSTM(self.h_dim, self.h_dim, self.n_layers, bias)  # , batch_first=True

    def forward(self, y):
        y, meta = self.scattering_transform(y)
        scattering_original = y
        loss = torch.zeros(1, device=self.device, requires_grad=True)

        # initialization
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

        def compute_prior_mean(prior_t_):
            return self.prior_mean(prior_t_) + self.r_prior_mean(prior_t_)

        def compute_prior_logvar(prior_t_):
            prior_logvar = self.prior_logvar(prior_t_) + self.r_prior_logvar(prior_t_)
            return self.logvar_acv(prior_logvar)

        def compute_enc(phi_y_t_, phi_h_t_):
            enc_t_ = (self.enc(torch.cat([phi_y_t_, phi_h_t_], 1)) +
                      self.r_enc(torch.cat([phi_y_t_, phi_h_t_], 1)))
            return enc_t_

        def compute_enc_mean(enc_t_):
            return self.enc_mean(enc_t_) + self.r_enc_mean(enc_t_)

        def compute_enc_logvar(enc_t_):
            enc_logvar_ = self.enc_logvar(enc_t_) + self.r_enc_logvar(enc_t_)
            return self.logvar_acv(enc_logvar_)

        def compute_phi_z(z_t_):
            return self.phi_z(z_t_) + self.r_phi_z(z_t_)

        def compute_dec(phi_z_t_):
            return self.dec(phi_z_t_) + self.r_dec(phi_z_t_)

        def compute_dec_mean(dec_t_):
            return self.dec_mean(dec_t_) + self.r_dec_mean(dec_t_)

        def compute_dec_logvar(dec_t_):
            dec_logvar_t_ = self.dec_logvar(dec_t_) + self.r_dec_logvar(dec_t_)
            return self.logvar_acv(dec_logvar_t_)

        # for all time steps
        for t in range(y.size(0)):
            # feature extraction: y_t
            y_t = y[t]
            h_t = h[-1]

            phi_y_t_future = torch.jit.fork(compute_phi_y_t, y_t)
            phi_h_t_future = torch.jit.fork(compute_phi_h_t, h_t)

            phi_y_t = torch.jit.wait(phi_y_t_future)
            phi_h_t = torch.jit.wait(phi_h_t_future)

            prior_t_future = torch.jit.fork(compute_prior_t, h_t)
            enc_t_future = torch.jit.fork(compute_enc, phi_y_t, phi_h_t)

            prior_t = torch.jit.wait(prior_t_future)
            enc_t = torch.jit.wait(enc_t_future)

            enc_mean_future = torch.jit.fork(compute_enc_mean, enc_t)
            enc_logvar_future = torch.jit.fork(compute_enc_logvar, enc_t)
            prior_mean_future = torch.jit.fork(compute_prior_mean, prior_t)
            prior_logvar_future = torch.jit.fork(compute_prior_logvar, prior_t)

            enc_mean_t = torch.jit.wait(enc_mean_future)
            enc_logvar_t = torch.jit.wait(enc_logvar_future)
            prior_mean_t = torch.jit.wait(prior_mean_future)
            prior_logvar_t = torch.jit.wait(prior_logvar_future)

            # sampling and re-parameterization: get a new z_t
            temp = tdist.Normal(enc_mean_t, enc_logvar_t.exp().sqrt())  # creates a normal distribution object
            z_t = tdist.Normal.rsample(temp)  # sampling from the distribution
            if self.modify_z is not None:
                modify_dims = self.modify_z.get('modify_dims')
                scale = self.modify_z.get('scale')
                shift = self.modify_z.get('shift')
                z_t = self._modify_z(z=z_t, modify_dims=modify_dims, scale=scale, shift=shift)

            # feature extraction: z_t
            phi_z_t = compute_phi_z(z_t)
            x_dec_t = compute_dec(phi_z_t)

            dec_mean_future = torch.jit.fork(compute_dec_mean, x_dec_t)
            dec_logvar_future = torch.jit.fork(compute_dec_logvar, x_dec_t)

            dec_mean_t = torch.jit.wait(dec_mean_future)
            dec_logvar_t = torch.jit.wait(dec_logvar_future)

            # recurrence: y_t, z_t -> h_t+1
            _, (h, c) = self.rnn(torch.cat([phi_y_t, phi_z_t], 1).unsqueeze(0), (h, c))

            if self.modify_h is not None:
                modify_dims = self.modify_h.get('modify_dims')
                scale = self.modify_h.get('scale')
                shift = self.modify_h.get('shift')
                h = self._modify_h(h=h, modify_dims=modify_dims, scale=scale, shift=shift)

            pred_dist = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt())
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

        torch_enc_mean = torch.stack(all_enc_mean, dim=2)
        torch_enc_std = torch.stack(all_enc_std, dim=2)
        torch_z = torch.stack(all_z_t, dim=2)
        torch_dec_std = torch.stack(all_dec_std, dim=2)
        torch_dec_mean = torch.stack(all_dec_mean, dim=2)
        torch_kld_values = torch.stack(all_kld, dim=2)
        torch_h = torch.stack(all_h, dim=2)
        results = VrnnForward(
            rec_loss=loss,  # (1,)
            kld_loss=kld_loss,  # ()
            nll_loss=loss,
            encoder_mean=torch_enc_mean,  # (batch_size, latent_dim)
            encoder_std=torch_enc_std,  # (batch_size, latent_dim)
            decoder_mean=torch_dec_mean,  # (batch_size, input_dim)
            decoder_std=torch_dec_std,  # (batch_size, input_dim)
            kld_values=torch_kld_values,  # (batch_size, latent_dim)
            sx=scattering_original,  # (input_size, batch_size, input_dim)
            z_latent=torch_z,  # (batch_size, latent_dim)
            hidden_states=torch_h   # (n_layers, batch_size, input_dim)
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
            prior_mean_logvar_t = self.prior_mean_logvar(prior_t)
            prior_mean_t = prior_mean_logvar_t[:, :self.z_dim]
            prior_logvar_t = self.logvar_acv(prior_mean_logvar_t[:, self.z_dim:])
            # prior_mean_t = self.prior_mean(prior_t)
            # prior_logvar_t = self.prior_logvar(prior_t)
            # sampling and reparameterization: get new z_t
            temp = tdist.Normal(prior_mean_t, prior_logvar_t.exp().sqrt())
            z_t = tdist.Normal.rsample(temp)

            # feature extraction: z_t
            phi_z_t = self.phi_z(z_t) + self.r_phi_z(z_t)

            dec_t = self.dec(phi_z_t) + self.r_dec(phi_z_t)
            dec_mean_logvar_t = self.dec_mean_logvar(dec_t)
            dec_mean_t = dec_mean_logvar_t[:, :self.input_dim]
            dec_logvar_t = self.logvar_acv(dec_mean_logvar_t[:, self.input_dim:])
            temp = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt())

            phi_y_t = self.phi_y(dec_mean_t) + self.r_phi_y(dec_mean_t)

            sample[t] = tdist.Normal.rsample(temp)
            sample_mu[t] = dec_mean_t
            sample_sigma[t] = dec_logvar_t
            _, (h, c) = self.rnn(torch.cat([phi_y_t, phi_z_t], 1).unsqueeze(0), (h, c))

        return sample, sample_mu, sample_sigma


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth_multiplier=1, activation=None):
        super(SeparableConv2D, self).__init__()
        # nn.ReLU()
        # filters: 1, depth_multiplier:1, kernel: (1*1)
        self.depthwise = nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size=kernel_size,
                                   groups=in_channels, padding='same')
        self.pointwise = nn.Conv2d(in_channels * depth_multiplier, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

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
        # Apply the convolutional layer
        # input is shape (batch, latent_dim, input_size)
        # x = x.permute(0, 2, 1).unsqueeze(2)  # change to (batch, channels, height, width) ->  (batch, 3, 1, 150)
        x = x.unsqueeze(2)
        x = self.conv_layer(x)  # (batch_size, conv_out_channels, input_size, input_size)

        # Reshape x to (batch_size, seq_len, features) before feeding to LSTM
        # batch_size, channels, height, width = x.size()
        # x = x.view(batch_size, channels * height, width)

        # Apply LSTM layers
        # x shape: (batch, latent_size, 1, 150)

        x = x.squeeze(2).permute(0, 2, 1)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)

        # shape could be (batch_size, seq_len, lstm_h[-1])

        # Take the last output of the LSTM
        # x = x[:, -1, :]

        # Apply the linear layer
        x = self.linear(x)

        # Apply the activation function
        # x = self.fc_activation(x)

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
