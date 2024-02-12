import yaml
import logging
from datetime import datetime
import os
import sys
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
from scipy.signal import decimate
from models.model import VAE, VAE_linear
import matplotlib.pyplot as plt
import numpy as np
from models.utils import plot_scattering, \
    plot_original_reconstructed
import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Utils ----------------------------------------------------------------------------------------------------------------
class StreamToLogger(object):
    """
    Stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def normalize_data(seq_list, min_val, max_val):
    # todo there could be a better way to do normalization in the pipline
    range_val = max_val - min_val
    for dict_item in seq_list:
        dict_item['fhr'] = [((x - min_val) / range_val) for x in dict_item['fhr']]


def prepare_data(file_path=None, do_decimate=True):
    with open(file_path, 'rb') as input_file:
        dict_list = pickle.load(input_file)
    if do_decimate:
        for dict_item in dict_list:
            dict_item['fhr'] = decimate(dict_item['fhr'], 16).tolist()
    return dict_list
# ----------------------------------------------------------------------------------------------------------------------


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    kld = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + (0.001 * kld)

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
    def __init__(self):
        super(VAELoss, self).__init__()

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

        # Scale KL divergence
        kl_div_weighted = 0.99 * kl_div

        # Total loss
        total_loss = recon_loss + kl_div_weighted

        return total_loss, recon_loss, kl_div


# Custom dataset classes for different types of records ----------------------------------------------------------------
class FHRDataset(Dataset):
    # todo how to get the GUID
    def __init__(self, list_dicts):
        # Concatenating 'fhr' from both lists
        self.data = [d['fhr'] for d in list_dicts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert data to PyTorch tensor
        sample = torch.tensor(self.data[idx], dtype=torch.float)
        return sample


#  Dataset class from numpy .npy data
class SignalDataset(Dataset):
    def __init__(self, data):
        # Assuming data is a NumPy array of shape (4000, 300)
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sample at the specified index
        sample = self.data[idx]
        return sample


class JsonDatasetPreload(Dataset):
    def __init__(self, json_folder_path):
        self.data_files = [os.path.join(json_folder_path, file) for file in os.listdir(json_folder_path) if
                           file.endswith('.json')]
        self.samples = []

        # Load data
        for file_path in self.data_files:
            print(file_path)
            with open(file_path, 'r') as file:
                data = json.load(file)
                # Assuming each file contains a single sample for simplicity
                self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_data = self.samples[idx]
        # Extracting the `fhr` data and possibly other information
        fhr = torch.tensor(sample_data['fhr'])
        up = torch.tensor(sample_data['up'])
        target = torch.tensor(sample_data['target'])
        sample_weight = torch.tensor(sample_data['sample_weights'])
        # return fhr, target, sample_weight
        return fhr


class JsonDataset(Dataset):
    def __init__(self, json_folder_path):
        self.data_files = [os.path.join(json_folder_path, file) for file in os.listdir(json_folder_path) if
                           file.endswith('.json')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load only the needed JSON file
        file_path = self.data_files[idx]
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract data
        fhr = torch.tensor(data['fhr'])
        up = torch.tensor(data['up'])
        target = torch.tensor(data['target'])
        sample_weight = torch.tensor(data['sample_weights'])

        return fhr, target, sample_weight


def train(model=None, optimizer=None, loss_fn=None, train_dataloader=None, train_plot_dir=None, epoch_num=None):
    train_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="training", leave=False)
    loss_train_per_batch = []
    model.train()
    for batch_idx, batch_data in train_iterator:
        batch_data = batch_data.to(device)
        batch_size = batch_data.size(0)
        seq_len = batch_data.size(1)

        original_x, reconstructed_x, mean, logvar = model(batch_data)
        total_loss, recon_loss, kl_div = loss_fn(reconstructed_x, original_x, mean, logvar)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        loss_train_per_batch.append(total_loss.item())
        train_iterator.set_postfix(loss=total_loss.item())
        if epoch_num > 500:
            if batch_idx % 100 == 0:
                # plot_scattering(signal=batch_data[0], Sx=original_x[0].permute(1, 0), Sxr=reconstructed_x[0],
                #                 do_plot_rec=True, tag=f'train_{epoch_num}_', plot_dir=train_plot_dir)
                plot_original_reconstructed(original_x=original_x[0].cpu().detach().numpy(),
                                            reconstructed_x=reconstructed_x[0].cpu().detach().numpy(),
                                            plot_dir=train_plot_dir, tag=f'train_{epoch_num}_{batch_idx}_')
    loss_train = np.mean(loss_train_per_batch)
    train_iterator.set_postfix(loss=loss_train)
    return loss_train


def test(model=None, loss_fn=None, valid_dataloader=None, validation_plot_dir=None, epoch_num=None):
    model.eval()
    valid_iterator = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), desc="testing")
    loss_test_per_batch = []
    with torch.no_grad():
        for batch_idx, batch_data in valid_iterator:
            batch_data = batch_data.to(device)
            original_x, reconstructed_x, mean, logvar = model(batch_data)
            total_loss, recon_loss, kl_div = loss_fn(reconstructed_x, original_x, mean, logvar)
            loss_test_per_batch.append(total_loss.item())
            if epoch_num > 500:
                if batch_idx % 100 == 0:
                    # recon_x = reconstructed_x.detach().cpu().numpy()
                    # true_fhr = batch_data.cpu().numpy()  # nd(64, 300)
                    indices = torch.randperm(original_x.size(0))[:5]
                    original_x_sampled = original_x[indices]
                    reconstructed_x_sampled = reconstructed_x[indices]
                    plt_c = 0
                    for i in range(original_x_sampled.size(0)):
                        original_x_selected = original_x_sampled[i]
                        reconstructed_x_selected = reconstructed_x_sampled[i]
                        # plot_scattering(signal=batch_data[0], Sx=original_x[0].permute(1, 0), Sxr=reconstructed_x[0],
                        #                 do_plot_rec=True, tag=f'test_{epoch_num}_{i}_', plot_dir=validation_plot_dir)
                        plot_original_reconstructed(original_x=original_x[i].cpu().detach().numpy(),
                                                    reconstructed_x=reconstructed_x[i].cpu().detach().numpy(),
                                                    plot_dir=validation_plot_dir, tag=f'test_{epoch_num}_{i}_')
                # predicted_fhr = reconstructed_x.cpu().numpy() # nd(64, 300)
                # selected_indices = np.random.choice(true_fhr.shape[0], 5, replace=False)
                # selected_signals_true = true_fhr[selected_indices, :]
                # selected_signals_pred = predicted_fhr[selected_indices, :]
                # plt.figure(figsize=(10, 8))
                # for j, signal in enumerate(selected_signals_true):
                #     plt.subplot(5, 1, j + 1)
                #     plt.plot(selected_signals_true[j], label="True Signal", linewidth=3)
                #     plt.plot(selected_signals_pred[j], label="Predicted Signal", linewidth=1)
                #     plt.legend(loc='upper right')
                #     plt.xlabel('Sample')
                #     plt.ylabel('FHR')
                # plt.tight_layout()
                # fig_path = os.path.join(validation_plot_dir, f'plots{epoch_num}_{plt_c}.png')
                # plt_c += 1
                # plt.savefig(fig_path)
                # plt.close()
    loss_test = np.mean(loss_test_per_batch)
    return loss_test


if __name__ == "__main__":
    # read config file -------------------------------------------------------------------------------------------------
    with open('./config_arguments.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    print('==' * 100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # setup configs ----------------------------------------------------------------------------------------------------

    # creating the corresponding folders and paths ---------------------------------------------------------------------
    now = datetime.now()
    run_date = now.strftime("%Y-%m-%d_%H-%M")
    output_base_dir = os.path.normpath(config['folders_config']['out_dir_base'])
    train_results_dir = os.path.join(output_base_dir, run_date, 'train_results')
    model_checkpoint_dir = os.path.join(output_base_dir, run_date, 'model_checkpoints')
    folders_list = [output_base_dir, train_results_dir, model_checkpoint_dir]
    for folder in folders_list:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # setting up the logging -------------------------------------------------------------------------------------------
    log_file = os.path.join(train_results_dir, 'log.txt')
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)

    # Preparing training and testing datasets --------------------------------------------------------------------------
    # todo improve this implementation for a more general dataset
    dataset_dir = os.path.normpath(config['dataset_config']['dataset_dir'])
    stat_path = os.path.normpath(config['dataset_config']['stat_path'])
    batch_size = config['general_config']['batch_size']['train']
    # healthy_dataset_path = os.path.join(dataset_dir, 'HEALTHY_signal_dicts.pkl')
    # hie_dataset_path = os.path.join(dataset_dir, 'HIE_signal_dicts.pkl')
    #
    # healthy_list = prepare_data(healthy_dataset_path, do_decimate=False)
    # hie_list = prepare_data(hie_dataset_path, do_decimate=False)

    # fhr_values = [dict_item['fhr'] for dict_item in healthy_list + hie_list]
    # min_fhr = min([min(fhr) for fhr in fhr_values])
    # max_fhr = max([max(fhr) for fhr in fhr_values])
    # normalize_data(healthy_list, min_fhr, max_fhr)
    # normalize_data(hie_list, min_fhr, max_fhr)
    fhr_healthy_dataset = JsonDatasetPreload(dataset_dir)
    # fhr_healthy_dataset = FHRDataset(healthy_list)
    dataset_size = len(fhr_healthy_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size

    # k_folds = 5
    # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    # for fold, (train_index, test_index) in enumerate(kf.split(fhr_healthy_dataset)):
    #     train_subsampler = Subset(fhr_healthy_dataset, train_index)
    #     test_subsampler = Subset(fhr_healthy_dataset, test_index)

    train_dataset, test_dataset = random_split(fhr_healthy_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f'Train size: {len(train_dataset)} \n Test size: {len(test_dataset)}')
    print('==' * 100)
    # fhr_hie_dataset = FHRDataset(hie_list[0:10])
    # hie_dataloader = DataLoader(fhr_hie_dataset, batch_size=1, shuffle=False)

    # define model and train it ----------------------------------------------------------------------------------------
    # todo a better way to implement model config
    input_size = config['model_config']['VAE_model']['input_size']
    input_dim = config['model_config']['VAE_model']['input_dim']
    latent_size = config['model_config']['VAE_model']['latent_size']
    num_LSTM_layers = config['model_config']['VAE_model']['num_LSTM_layers']
    dec_hidden_dim = config['model_config']['VAE_model']['decoder_hidden_dim']
    enc_hidden_dim = config['model_config']['VAE_model']['encoder_hidden_dim']

    max_iter = config['general_config']['max_iter']
    lr = config['general_config']['lr']
    # model ------------------------------------------------------------------------------------------------------------
    # VAE_model = VAE(
    #     input_size=input_size,
    #     input_dim=input_dim,
    #     dec_hidden_dim=dec_hidden_dim,
    #     enc_hidden_dim=enc_hidden_dim,
    #     latent_size=latent_size,
    #     num_LSTM_layers=num_LSTM_layers
    # )
    VAE_model = VAE_linear(input_seq_size=300, latent_dim=120)
    print(f'Model:  \n {VAE_model}')
    print('==' * 100)
    VAE_model.to(device)
    params = VAE_model.parameters()
    trainable_params = sum(p.numel() for p in VAE_model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable_params}')
    print('==' * 100)
    optimizer = torch.optim.Adam(VAE_model.parameters(), lr=lr, weight_decay=0)
    epochs = tqdm(range(max_iter // len(train_loader) + 1))
    vae_loss_re = VAELoss()
    train_loss_list = []
    test_loss_list = []
    for epoch in epochs:
        loss_train = train(
            model=VAE_model,
            optimizer=optimizer,
            loss_fn=vae_loss_re,
            train_dataloader=train_loader,
            train_plot_dir=train_results_dir,
            epoch_num=epoch
        )

        train_loss_list.append(loss_train)

        loss_test = test(
            model=VAE_model,
            loss_fn=vae_loss_re,
            valid_dataloader=test_loader,
            validation_plot_dir=train_results_dir,
            epoch_num=epoch
        )

        test_loss_list.append(loss_test)

    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 32
    plt.rcParams['text.usetex'] = True
    loss_dict = {'train': train_loss_list, 'test': test_loss_list}
    loss_path = os.path.join(train_results_dir, 'loss_dict.pkl')
    with open(loss_path, 'wb') as file:
        pickle.dump(loss_dict, file)
    t = np.arange(1, len(epochs) + 1)
    ax[0].autoscale(enable=True, axis='x', tight=True)
    ax[0].plot(t, train_loss_list, label='train loss', color='#FF5733', linewidth=1)  # Custom hex color and line thickness
    ax[1].autoscale(enable=True, axis='x', tight=True)
    ax[1].plot(t, test_loss_list, label='test loss', color='#005B41', linewidth=1)  # Custom hex color and line thickness
    # Adding grid, legend, and labels with specific requirements
    # ax[0].grid(True)
    # ax[0].legend()
    # ax[0].set_xlabel('$\ell_2$')
    plt.savefig(f'{train_results_dir}/Loss_st.pdf', bbox_inches='tight')
