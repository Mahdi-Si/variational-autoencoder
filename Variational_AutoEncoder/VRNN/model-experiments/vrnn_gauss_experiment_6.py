

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
            nn.Linear(self.input_dim, 11),
            nn.LayerNorm(11),
            nn.ELU(),
            nn.Linear(11, 11),
            nn.LayerNorm(11),
            nn.ELU(),
            nn.Linear(11, 11)

        )

        self.r_phi_y = nn.Linear(self.input_dim, 11)

        self.phi_h = nn.Sequential(
            nn.Linear(self.h_dim, 18),
            nn.LayerNorm(18),
            nn.ELU(),
            nn.Linear(18, 14),
            nn.LayerNorm(14),
            nn.ELU(),
            nn.Linear(14, 11),
            nn.LayerNorm(11),
            nn.ELU(),
        )

        self.r_phi_h = nn.Sequential(
            nn.Linear(self.h_dim, int(self.h_dim / 2))
        )

        self.prior = nn.Sequential(
            nn.Linear(self.h_dim, 18),
            nn.LayerNorm(18),
            nn.ELU(),
            nn.Linear(18, 14),
            nn.LayerNorm(14),
            nn.ELU(),
            nn.Linear(14, 11),
            nn.LayerNorm(11),
            nn.ELU(),
        )

        self.r_prior = nn.Sequential(
            nn.Linear(self.h_dim, int(self.h_dim / 2))
        )

        self.prior_mean_logvar = nn.Sequential(
            nn.Linear(int(self.h_dim / 2), 2 * 9),
            nn.LayerNorm(2 * 9),
            nn.ELU(),
            nn.Linear(2 * 9, 2 * 6),
            nn.LayerNorm(2 * 6),
            nn.ELU(),
            nn.Linear(2 * 6, 2 * 3),
        )

        self.prior_logvar_activation = nn.Softplus()

        # encoder function (phi_enc) -> Inference
        self.enc = nn.Sequential(
            nn.Linear(2 * int(self.h_dim / 2), 18),
            nn.LayerNorm(18),
            nn.ELU(),
            nn.Linear(18, 14),
            nn.LayerNorm(14),
            nn.ELU(),
            nn.Linear(14, 11),
        )

        self.r_enc = nn.Sequential(
            nn.Linear(self.h_dim, int(self.h_dim / 2))
        )

        self.enc_mean_logvar = nn.Sequential(
            nn.Linear(int(self.h_dim / 2), 2 * 9),
            nn.LayerNorm(2 * 9),
            nn.ELU(),
            nn.Linear(2 * 9, 2 * 6),
            nn.LayerNorm(2 * 6),
            nn.ELU(),
            nn.Linear(2 * 6, 2 * 3),
        )

        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, 6),
            nn.LayerNorm(6),
            nn.ELU(),
            nn.Linear(6, 9),
            nn.LayerNorm(9),
            nn.ELU(),
            nn.Linear(9, 11)
        )

        self.r_phi_z = nn.Sequential(
            nn.Linear(self.z_dim, int(self.h_dim / 2))
        )

        # decoder function (phi_dec) -> Generation
        self.dec = nn.Sequential(
            nn.Linear(11, 11),
            nn.LayerNorm(11),
            nn.ELU(),
            nn.Linear(11, 11),
            nn.LayerNorm(11),
            nn.ELU(),
            nn.Linear(11, 11),
        )

        self.r_dec = nn.Sequential(
            nn.Linear(int(self.h_dim / 2), 11)
        )

        self.dec_mean_logvar = nn.Sequential(
            nn.Linear(11, 2 * 11),
            nn.LayerNorm(2 * 11),
            nn.ELU(),
            nn.Linear(2 * 11, 2 * 11),
            nn.LayerNorm(2 * 11),
            nn.ELU(),
            nn.Linear(2 * 11, 2 * 11),
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

        # for all time steps
        for t in range(y.size(0)):
            # feature extraction: y_t
            # y original is shape (batch_size, input_size, input_dim)
            phi_y_t = self.phi_y(y[t]) + self.r_phi_y(y[t])

            phi_h_t = self.phi_h(h[-1]) + self.r_phi_h(h[-1])

            prior_t = self.prior(h[-1]) + self.r_prior(h[-1])
            prior_mean_logvar_t = self.prior_mean_logvar(prior_t)
            prior_mean_t = prior_mean_logvar_t[:, :self.z_dim]
            prior_logvar_t = self.logvar_acv(prior_mean_logvar_t[:, self.z_dim:])
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
            dec_mean_logvar_t = self.dec_mean_logvar(dec_t)
            dec_mean_t = dec_mean_logvar_t[:, :self.input_dim]
            dec_logvar_t = self.logvar_acv(dec_mean_logvar_t[:, self.input_dim:])
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
