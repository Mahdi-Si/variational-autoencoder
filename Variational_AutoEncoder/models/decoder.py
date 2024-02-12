import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvLinDecoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.linear2 = nn.Linear(90, 100)
        self.linear1 = nn.Linear(100, 152)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 8, 19))

        # Inverse of the encoder's convolutional layers
        self.convT3 = nn.ConvTranspose1d(8, 4, 3, stride=2, padding=0, output_padding=1)  # Adjust kernel size, stride, padding if necessary
        # self.batch2 = nn.BatchNorm1d(2)
        self.convT2 = nn.ConvTranspose1d(4, 2, 3, stride=2, padding=0, output_padding=1)
        self.convT1 = nn.ConvTranspose1d(2, 1, 3, stride=2, padding=0, output_padding=1)
        self.linear_out = nn.Linear(166, 150)

    def forward(self, x):
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear1(x))
        x = x.view(-1, 8, 19)
        # x = self.unflatten(x)
        x = F.relu(self.convT3(x))
        # x = F.relu(self.batch2(self.convT2(x)))
        x = F.relu((self.convT2(x)))
        x = F.relu(self.convT1(x))
        x = F.relu(self.linear_out(x))
        # Example: reshape to match expected input of convT3, adjust 37 accordingly
        return x


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim=None, hidden_size=None, hidden_sizes=None, num_layers=None, output_dim=None,
                 sequence_length=None):
        """
        LSTM Decoder
        :param latent_dim: The dimensionality of the latent space.
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers.
        :param output_dim: The number of expected features in the output x, matching the input_dim of the encoder.
        :param sequence_length: The length of the output sequence, matching the sequence_length of the encoder input.
        """
        super(LSTMDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Mapping the latent space back to a dimension that can be reshaped into the expected LSTM input shape
        self.initial_linear = nn.Sequential(
            nn.Linear(150, 200),
            nn.ReLU(),
            nn.Linear(200, 800),
            nn.ReLU(),
            nn.Linear(800, 1500),
        )

        # Assuming the encoder's LSTM output is batch normalized, we might not necessarily reverse that operation here
        # Batch normalization in decoders can be tricky due to the batch statistics behaving differently during training and inference

        # Recreate the LSTM structure from the encoder
        # self.lstm = nn.LSTM(1500 // sequence_length, hidden_size, num_layers, batch_first=True)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM((1500 // sequence_length) if i == 0 else hidden_sizes[i - 1], hidden_sizes[i], batch_first=True)
            for i in range(len(hidden_sizes))
        ])

        # Final layer to match the output dimension to the original input dimension
        # self.final_linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        """
        Forward pass
        :param x: input from latent space (batch_size, latent_dim)
        :return: reconstructed time series data (batch_size, sequence_length, output_dim)
        """
        # Map the latent vectors back to a suitable dimension
        x = self.initial_linear(x)

        # Reshape to match the LSTM input shape
        x = x.view(-1, self.sequence_length, 1500 // self.sequence_length)

        lstm_out = x
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)
        # Apply LSTM
        # output, (hidden, cell) = self.lstm(x)

        # Final layer to produce the reconstructed sequence
        # output = self.final_linear(output)

        return lstm_out


# class LSTMDecoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, sequence_length):
#         """
#         LSTM Decoder
#         :param latent_dim: Dimensionality of the latent space
#         :param hidden_size: Number of features in the hidden state of the LSTM
#         :param num_layers: Number of recurrent layers
#         :param output_size: The number of features in the output x (same as the input size of the encoder)
#         :param sequence_length: Length of the output sequence to be generated
#         """
#         super(LSTMDecoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.sequence_length = sequence_length
#         self.input_size = input_size
#
#         # Initial linear layer to map latent space to features
#         # self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)
#
#         # LSTM layers
#         self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
#                             batch_first=True)
#
#         # Output linear layer
#         # self.Linear_output = nn.Linear(self.hidden_size * self.hidden_size, self.sequence_length)
#
#     def forward(self, z, h_=None):
#         """
#         Forward pass
#         :param z: Latent space representation (batch_size, latent_dim)
#         :return: Reconstructed time-series data (batch_size, sequence_length, output_size)
#         """
#         # Map latent vector z to the hidden state size
#         # z = z.unsqueeze(-1)
#         # z = self.hidden_to_output(z)
#         # hidden = self.latent_to_hidden(z).unsqueeze(1).repeat(1, self.sequence_length, 1)
#         # Generate sequence
#         # z = z.unsqueeze(-1)
#         # lstm_out, _ = self.lstm(z, h_)
#         lstm_out, _ = self.lstm(z)
#         # lstm_out_ = torch.flatten(lstm_out, start_dim=1)
#         # Map LSTM outputs to desired output size
#         # output = self.Linear_output(lstm_out_)
#         return lstm_out


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels, sequence_length):
        """
        1D CNN Decoder
        :param latent_dim: Dimensionality of the latent space
        :param output_channels: Number of channels in the output (same as the input channels of the encoder)
        :param sequence_length: Length of the output sequence (same as the input sequence length of the encoder)
        """
        super(CNNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        # Define the starting size
        self.start_size = sequence_length // 8  # Adjust depending on the number of convolutional layers in encoder
        self.fc = nn.Linear(latent_dim, 64 * self.start_size)

        # Define transposed convolutional layers
        self.conv1 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose1d(16, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        """
        Forward pass
        :param z: Latent space representation (batch_size, latent_dim)
        :return: Reconstructed time-series data (batch_size, output_channels, sequence_length)
        """
        # Map latent vector z to the sequence
        x = self.fc(z)
        x = x.view(-1, 64, self.start_size)  # Reshape to (batch_size, channels, length)

        # Apply transposed convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv3(x))  # Tanh activation for the last layer

        return x


class GRUDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, num_layers, output_size, sequence_length):
        """
        GRU Decoder
        :param latent_dim: Dimensionality of the latent space
        :param hidden_size: Number of features in the hidden state of the GRU
        :param num_layers: Number of recurrent layers
        :param output_size: Number of features in the output (same as the input size of the encoder)
        :param sequence_length: Length of the output sequence to be generated
        """
        super(GRUDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # Map latent vector to hidden state size
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)

        # GRU layers
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        # Output layer
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        """
        Forward pass
        :param z: Latent space representation (batch_size, latent_dim)
        :return: Reconstructed time-series data (batch_size, sequence_length, output_size)
        """
        # Prepare initial hidden state from latent vector
        hidden_state = self.latent_to_hidden(z).unsqueeze(0).repeat(self.num_layers, 1, 1)

        # Prepare a dummy input for the first time step
        dummy_input = torch.zeros(z.size(0), 1, self.hidden_size, device=z.device)

        # Generate sequence
        outputs = []
        for _ in range(self.sequence_length):
            dummy_input, hidden_state = self.gru(dummy_input, hidden_state)
            output = self.hidden_to_output(dummy_input.squeeze(1))
            outputs.append(output.unsqueeze(1))

        # Concatenate outputs for each time step
        outputs = torch.cat(outputs, dim=1)

        return outputs


# FIXME: experimental transformer decoder

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, d_model, nhead, num_decoder_layers, dim_feedforward, output_size, sequence_length):
        """
        Transformer Decoder
        :param latent_dim: Dimensionality of the latent space
        :param d_model: The size of the embedding (also used as the model size in the transformer)
        :param nhead: The number of heads in the multiheadattention model
        :param num_decoder_layers: The number of sub-decoder-layers in the decoder
        :param dim_feedforward: The dimension of the feedforward network model
        :param output_size: Number of features in the output
        :param sequence_length: Length of the output sequence
        """
        super(TransformerDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Initial linear layer to map latent space to features
        self.latent_to_features = nn.Linear(latent_dim, sequence_length * d_model)

        # Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output linear layer
        self.features_to_output = nn.Linear(d_model, output_size)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, z):
        """
        Forward pass
        :param z: Latent space representation (batch_size, latent_dim)
        :return: Reconstructed time-series data (batch_size, sequence_length, output_size)
        """
        # Map latent vector to feature space
        x = self.latent_to_features(z)
        x = x.view(-1, self.sequence_length, self.d_model)

        # Apply positional encoding
        x = self.pos_encoder(x)

        # Generate dummy target sequence for Transformer Decoder
        target_sequence = torch.zeros_like(x)

        # Decode
        output = self.transformer_decoder(tgt=target_sequence, memory=x)
        output = self.features_to_output(output)

        return output

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
