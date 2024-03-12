import torch
from torch import nn
from torch.nn import functional as F
import math
from kymatio.torch import Scattering1D


class ConvLinEncoder(nn.Module):
    def __init__(self, seq_len, latent_dim):
        super(ConvLinEncoder, self).__init__()
        self.linear_input = nn.Linear(150, 150)
        self.conv1 = nn.Conv1d(1, 2, 10, stride=2, padding=1)
        self.conv2 = nn.Conv1d(2, 4, 3, stride=2, padding=1)
        # self.batch2 = nn.BatchNorm1d(4)
        self.conv3 = nn.Conv1d(4, 8, 3, stride=2, padding=1)
        self.linear1 = nn.Linear(144, 100)  # todo check this
        self.linear2 = nn.Linear(100, 70)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.elu(self.linear_input(x))
        x = F.elu(self.conv1(x))
        # x = F.relu(self.batch2(self.conv2(x)))
        x = F.elu((self.conv2(x)))
        x = F.elu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.elu(self.linear1(x))
        x = self.dropout(x)
        x = F.elu(self.linear2(x))
        return x


class ConvEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        LSTM Encoder
        :param input_dim: The number of expected features in the input x
        :param hidden_dim: The number of features in the hidden state h
        :param latent_dim: Number of recurrent layers.
        """
        super(ConvEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        # Calculate the size of the output from the conv layers to input into the LSTM
        self._to_lstm_dim = (hidden_dim * 2) * (input_dim // 4)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self._to_lstm_dim, hidden_size=hidden_dim, batch_first=True)

        # Linear layer
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten the output for the LSTM layer
        x = x.view(x.size(0), -1)

        # LSTM layer - taking the last hidden state
        _, (hidden, _) = self.lstm(x.unsqueeze(1))
        hidden = hidden[-1]
        # latent_vec = self.fc1(hidden)
        latent_vec = hidden

        return latent_vec


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dims, latent_size, latent_dim):
        """
        LSTM Encoder
        :param input_dim: The number of expected features in the input x
        :param hidden_dims: The number of features in the hidden state h
        :param latent_dim: The dimensionality of the latent space.
        """
        self.input_dim = input_dim
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.latent_size = latent_size
        self.latent_dim = latent_dim
        super(LSTMEncoder, self).__init__()
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_dim if i == 0 else self.hidden_dims[i - 1], self.hidden_dims[i], batch_first=True)
            for i in range(len(self.hidden_dims))
        ])
        # self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        # self.batch_norm = nn.BatchNorm1d(5)
        self.sequential = nn.Sequential(
            nn.Linear(self.input_size*self.hidden_dims[-1], self.latent_size*self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_size*self.latent_dim, self.latent_size*self.latent_dim),
            nn.Tanh(),
        )
        self.fc_mean = nn.Linear(self.latent_size*self.latent_dim, self.latent_size*self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_size * self.latent_dim, self.latent_size * self.latent_dim)
    def forward(self, x):
        """
        Forward pass
        :param x: input time series data (batch_size, sequence_length, input_size)
        :return: latent space mean and log variance
        """
        # lstm_out, (hidden, cells) = self.lstm_layers(x)
        lstm_out = x.permute(0, 2, 1)
        hidden_states = []
        # for i, lstm in enumerate(self.lstm_layers):
        #     if i == 0:
        #         h_0 = torch.zeros(1, lstm_out.size(0), self.hidden_dims[i]).to(lstm_out.device)
        #         c_0 = torch.zeros(1, lstm_out.size(0), self.hidden_dims[i]).to(lstm_out.device)
        #     else:
        #         h_0, c_0 = hidden_states[-1]
        #     lstm_out, (h_n, c_n) = lstm(lstm_out, h_0=h_0, c_0=c_0)
        #     hidden_states.append((h_n, c_n))
        for lstm in self.lstm_layers:
            lstm_out, (h_n, c_n) = lstm(lstm_out)
        # lstm_out shape (batch_size, signal_len, last lstm layer hidden_dim)
        # output = self.batch_norm(lstm_out.permute(0, 2, 1))
        output = torch.flatten(lstm_out, start_dim=1)  # Flatten shape (bath_size, signal_len * hidden_dim)
        output = self.sequential(output)
        mean_ = self.fc_mean(output)
        mean_ = mean_.view(-1, self.latent_size, self.latent_dim)
        logvar_ = self.fc_logvar(output)
        logvar_ = logvar_.view(-1, self.latent_size, self.latent_dim)
        # output_ = output.view(-1, self.latent_size, self.latent_dim)
        # output_ = self.sequential(output)
        return mean_, logvar_


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        """
        GRU Encoder
        :param input_size: Number of expected features in the input x
        :param hidden_size: Number of features in the hidden state
        :param num_layers: Number of recurrent layers
        :param latent_dim: Dimensionality of the latent space
        """
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # Mapping to latent space
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)


    def forward(self, x):
        """
        Forward pass
        :param x: Input time series data (batch_size, sequence_length, input_size)
        :return: Latent space mean and log variance
        """
        # GRU output
        output, hidden = self.gru(x)
        # Use only the last hidden state
        hidden = hidden[-1]
        # latent_vec = self.fc1(hidden)
        latent_vec = hidden
        return latent_vec


# FIXME: This is experimental code to implement transformer as encoder for VAE
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, latent_dim):
        """
        Transformer Encoder
        :param input_size: Number of expected features in the input (feature size)
        :param d_model: The size of the embedding (also used as the model size in the transformer)
        :param nhead: The number of heads in the multiheadattention models
        :param num_encoder_layers: The number of sub-encoder-layers in the encoder
        :param dim_feedforward: The dimension of the feedforward network model
        :param latent_dim: Dimensionality of the latent space
        """
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc_mean = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, src):
        """
        Forward pass
        :param src: Input time series data (batch_size, sequence_length, input_size)
        :return: Latent space mean and log variance
        """
        src = self.embedding(src)  # Embed input sequence
        src = self.pos_encoder(src)  # Apply positional encoding
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Aggregate over sequence
        mean = self.fc_mean(output)
        logvar = self.fc_logvar(output)
        return mean, logvar


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class CNNEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        """
        1D CNN Encoder
        :param input_channels: Number of channels (features) in the input
        :param latent_dim: Dimensionality of the latent space
        """
        super(CNNEncoder, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)

        # Define fully connected layers for mean and log variance
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        """
        Forward pass
        :param x: Input time series data (batch_size, channels, sequence_length)
        :return: Latent space mean and log variance
        """
        # Apply convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Produce mean and log variance for latent space
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class ScatteringNet(nn.Module):
    def __init__(self, J, Q, T, shape):
        super(ScatteringNet, self).__init__()
        self.scat = Scattering1D(J=J, Q=Q, T=T, shape=shape)

    def forward(self, x):
        # x = x.permute(0, 2, 1)  # Equivalent to Permute in TensorFlow
        x = x.unsqueeze(0)
        x = x.contiguous()
        return self.scat(x)

    def meta(self):
        return self.scat.meta()
