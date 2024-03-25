import torch.nn as nn
from kymatio.torch import Scattering1D
import torch
import torch.nn.functional as F


class EncoderProjection(nn.Module):
    def __init__(self, seq_len, hidden_dim, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            # nn.ELU(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ELU(),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim))

    def forward(self, x):
        return self.model(x)


class DecoderProjection(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, seq_len),
            nn.ReLU(),
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Linear(seq_len, seq_len))

    def forward(self, x):
        return self.model(x)


class ScatteringNet(nn.Module):
    def __init__(self, J, Q, T, shape):
        super(ScatteringNet, self).__init__()
        self.scat = Scattering1D(J=J, Q=Q, T=T, shape=shape)

    def forward(self, x):
        # x = x.permute(0, 2, 1)  # Equivalent to Permute in TensorFlow
        # x_tensor = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(2)
        x = x.permute(0, 2, 1)

        x = x.contiguous()
        return self.scat(x)

    def meta(self):
        return self.scat.meta()


def vae_loss(reconstructed_x, original_x, mean, logvar):
    """
    Compute the VAE loss function.
    :param reconstructed_x: the output from the decoder
    :param original_x: the original input data
    :param mean: the mean of the latent space distribution
    :param logvar: the log variance of the latent space distribution
    :return: total loss, reconstruction loss, KL divergence
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructed_x, original_x, reduction='sum')

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + (0.99 * kl_div)

    return total_loss, recon_loss, kl_div


class VAELoss(nn.Module):
    def __init__(self, beta=1):
        super(VAELoss, self).__init__()
        self.beta = beta

    def forward(self, reconstructed_x, original_x, mean, logvar):
        """
        Compute the VAE loss function.
        :param reconstructed_x: the output from the decoder
        :param original_x: the original input data
        :param mean: the mean of the latent space distribution
        :param logvar: the log variance of the latent space distribution
        :return: total loss, reconstruction loss, KL divergence
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed_x, original_x, reduction='sum')

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_div

        return total_loss, recon_loss, self.beta*kl_div
