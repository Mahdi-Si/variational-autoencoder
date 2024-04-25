import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from Variational_AutoEncoder.models.dataset_transform import ScatteringTransform
from abc import ABC, abstractmethod


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


class VrnnGaussAbs(nn.Module, ABC):
    def __init__(self, input_dim, input_size, h_dim, z_dim, n_layers, device, log_stat, modify_z=None, modify_h=None,
                 bias=False):
        super(VrnnGaussAbs, self).__init__()

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

    @abstractmethod
    def generate(self, u):
        pass

    @staticmethod
    def kld_gaussian(mean_1, logvar_1, mean_2, logvar_2):
        """
        Calculate the Kullback-Leibler divergence between two Gaussians with parameters
        mean_1, logvar_1 and mean_2, logvar_2.

        Parameters:
        mean_1 (torch.Tensor): Mean vector of the first Gaussian (batch_size, input_dim)
        logvar_1 (torch.Tensor): Log variance vector of the first Gaussian (batch_size, input_dim)
        mean_2 (torch.Tensor): Mean vector of the second Gaussian (batch_size, input_dim)
        logvar_2 (torch.Tensor): Log variance vector of the second Gaussian (batch_size, input_dim)

        Returns:
        torch.Tensor: Kullback-Leibler divergence for each batch element (batch_size,)
        """
        # Compute the term-by-term differences and sums needed for the KLD formula
        term1 = torch.exp(logvar_2) / torch.exp(logvar_1)
        term2 = (mean_2 - mean_1) ** 2 / torch.exp(logvar_1)
        term3 = logvar_1 - logvar_2
        kld_element = term1 + term2 - 1 + term3
        # Sum over the input dimensions and calculate the KLD for each batch element
        kld = 0.5 * (torch.sum(term1 + term2 - 1 + term3))

        return kld, kld_element

    @staticmethod
    def kld_gauss_(mu_q, logvar_q, mu_p, logvar_p):
        # Goal: Minimize KL divergence between q_pi(z|xi) || p(z|xi)
        # This is equivalent to maximizing the ELBO: - D_KL(q_phi(z|xi) || p(z)) + Reconstruction term
        # This is equivalent to minimizing D_KL(q_phi(z|xi) || p(z))
        term1 = logvar_p - logvar_q - 1
        term2 = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        final_term = term1 + term2
        kld = 0.5 * torch.sum(final_term)
        return kld, final_term

    @staticmethod
    def kld_gauss(mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        std_2 = torch.clamp(std_2, min=1e-9)
        std_1 = torch.clamp(std_1, min=1e-9)

        kld_element = (torch.log(std_2.pow(2) / std_1.pow(2)) - 1 +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2))
        #  kld_element -> tensor (batch_size, latent_dim)
        return 0.5 * torch.sum(kld_element), kld_element

    def init_rnn_output(self, batch_size, seq_len):
        phi_h_t = torch.zeros(batch_size, seq_len, self.h_dim).to(self.device)

        return phi_h_t

    # @staticmethod
    # def _modify_z(z, modify_dims, shift, scale):
    #     for i in modify_dims:
    #         z[:, i] = scale[i] * z[:, i] + shift[i]
    #     return z

    @staticmethod
    # todo make it same is _modify_h
    def _modify_z(z, modify_dims, shift, scale):
        for index, dim in enumerate(modify_dims):
            z[:, dim] = scale[index] * z[:, dim] + shift[index]
        return z

    @staticmethod
    def _modify_h(h, modify_dims, shift, scale):
        if h.dim() == 3:
            for index, dim in enumerate(modify_dims):
                h[:, :, dim] = scale[index] * h[:, :, dim] + shift[index]
        elif h.dim() == 2:
            for index, dim in enumerate(modify_dims):
                h[:, dim] = scale[index] * h[:, dim] + shift[index]
        return h
