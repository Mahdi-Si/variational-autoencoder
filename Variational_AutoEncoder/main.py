import yaml
import logging
from datetime import datetime
import sys
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
from datasets.custom_datasets import JsonDatasetPreload
from Variational_AutoEncoder.models.model import VAE_linear, VAE
import matplotlib.pyplot as plt
import numpy as np
from Variational_AutoEncoder.utils.data_utils import plot_original_reconstructed
import os
from models.misc import VAELoss
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np
import builtins

from Variational_AutoEncoder.datasets.custom_datasets import JsonDatasetPreload
from Variational_AutoEncoder.utils.data_utils import plot_scattering, plot_original_reconstructed, \
    calculate_stats, plot_scattering_v2, plot_loss_dict
from Variational_AutoEncoder.utils.run_utils import log_resource_usage

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Utils ----------------------------------------------------------------------------------------------------------------
class StreamToLogger:
    """
    Stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger_, log_level=logging.INFO):
        self.logger = logger_
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


# Setup your logging
def setup_logging(log_file_setup=None):
    # log_file = os.path.join(log_dir, 'log.txt')
    logger_s = logging.getLogger('my_app')
    logger_s.setLevel(logging.INFO)

    # Create handlers for both file and console
    file_handler = logging.FileHandler(log_file_setup, mode='w')
    console_handler = logging.StreamHandler()

    # Optional: add a formatter to include more details
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('- %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger_s.addHandler(file_handler)
    logger_s.addHandler(console_handler)

    return logger_s

# ----------------------------------------------------------------------------------------------------------------------


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    kld = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + (0.001 * kld)


# Function to update the learning rate of the optimizer
def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Todo implement log_stat for config and here
def train(model=None, optimizer=None, loss_fn=None, train_dataloader=None, train_plot_dir=None, epoch_num=None):
    for param_group in optimizer.param_groups:
        current_learning_rate = param_group['lr']
        print(f'Learning Rate; {current_learning_rate}')

    train_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="training", leave=False)
    loss_train_per_batch = []
    model.train()
    for batch_idx, batch_data in train_iterator:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        scattering_original, x_rec, z, mean, logvar = model(batch_data)
        total_loss, recon_loss, kl_div = loss_fn(x_rec, scattering_original, mean, logvar)
        total_loss.backward()
        optimizer.step()
        loss_train_per_batch.append(total_loss.item())
        train_iterator.set_postfix(loss=total_loss.item())
        if (epoch_num > 0) or (epochs_num == 10):
            if batch_idx % 100 == 0:
                idx = np.random.randint(0, len(scattering_original)-1, 1)
                # plot_scattering(signal=batch_data[0], Sx=original_x[0].permute(1, 0), Sxr=reconstructed_x[0],
                #                 do_plot_rec=True, tag=f'train_{epoch_num}_', plot_dir=train_plot_dir)
                # plot_original_reconstructed(original_x=original_x[idx[0]].cpu().detach().numpy(),
                #                             reconstructed_x=reconstructed_x[idx[0]].cpu().detach().numpy(),
                #                             plot_dir=train_plot_dir, tag=f'train_{epoch_num}_{batch_idx}_')
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
                    selected_idx = np.random.randint(0, len(original_x)-1, 5)
                    for i in selected_idx:
                        # original_x_selected = original_x_sampled[i]
                        # reconstructed_x_selected = reconstructed_x_sampled[i]
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
    config_file_path = r'config_arguments.yaml'
    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    print('==' * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # setup configs ----------------------------------------------------------------------------------------------------

    # creating the corresponding folders and paths ---------------------------------------------------------------------
    now = datetime.now()
    run_date = now.strftime("%Y-%m-%d---[%H-%M]")
    experiment_tag = config['general_config']['tag']
    output_base_dir = os.path.normpath(config['folders_config']['out_dir_base'])
    base_folder = f'{run_date}-{experiment_tag}'
    train_results_dir = os.path.join(output_base_dir, base_folder, 'train_results')
    test_results_dir = os.path.join(output_base_dir, base_folder, 'test_results')
    model_checkpoint_dir = os.path.join(output_base_dir, base_folder, 'model_checkpoints')
    aux_dir = os.path.join(output_base_dir, base_folder, 'aux_test_HIE')
    tensorboard_dir = os.path.join(output_base_dir, base_folder, 'tensorboard_log')
    folders_list = [output_base_dir, train_results_dir, test_results_dir, aux_dir, model_checkpoint_dir,
                    tensorboard_dir]
    for folder in folders_list:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # setting up the logging -------------------------------------------------------------------------------------------
    log_file = os.path.join(train_results_dir, 'log.txt')
    logger = setup_logging(log_file_setup=log_file)
    sys.stdout = StreamToLogger(logger, logging.INFO)

    # print yaml file properly -----------------------------------------------------------------------------------------
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    print('==' * 50)

    # Preparing training and testing datasets --------------------------------------------------------------------------
    dataset_dir = os.path.normpath(config['dataset_config']['dataset_dir'])
    stat_path = os.path.normpath(config['dataset_config']['stat_path'])
    batch_size = config['general_config']['batch_size']['train']
    aux_dataset_hie_dir = os.path.normpath(config['dataset_config']['aux_dataset_dir'])
    plot_every_epoch = config['general_config']['plot_frequency']
    epochs_num = config['general_config']['epochs']

    # healthy_dataset_path = os.path.join(dataset_dir, 'HEALTHY_signal_dicts.pkl')
    # hie_dataset_path = os.path.join(dataset_dir, 'HIE_signal_dicts.pkl')

    # healthy_list = prepare_data(healthy_dataset_path, do_decimate=False)
    # hie_list = prepare_data(hie_dataset_path, do_decimate=False)

    # fhr_values = [dict_item['fhr'] for dict_item in healthy_list + hie_list]
    # min_fhr = min([min(fhr) for fhr in fhr_values])
    # max_fhr = max([max(fhr) for fhr in fhr_values])
    # normalize_data(healthy_list, min_fhr, max_fhr)
    # normalize_data(hie_list, min_fhr, max_fhr)
    # fhr_healthy_dataset = FHRDataset(healthy_list)

    fhr_healthy_dataset = JsonDatasetPreload(dataset_dir)
    fhr_aux_hie_dataset = JsonDatasetPreload(aux_dataset_hie_dir)
    data_loader_complete = DataLoader(fhr_healthy_dataset, batch_size=batch_size, shuffle=False, num_workers=14)

    with open(stat_path, 'rb') as f:
        x_mean = np.load(f)
        x_std = np.load(f)
    log_stat = (x_mean, x_std)

    dataset_size = len(fhr_healthy_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size

    print(f'Train size: {train_size} \n Test size: {test_size}')
    # k_folds = 5
    # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    # for fold, (train_index, test_index) in enumerate(kf.split(fhr_healthy_dataset)):
    #     train_subsampler = Subset(fhr_healthy_dataset, train_index)
    #     test_subsampler = Subset(fhr_healthy_dataset, test_index)

    train_dataset, test_dataset = random_split(fhr_healthy_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    aux_hie_loader = DataLoader(fhr_aux_hie_dataset, batch_size=256, shuffle=False)
    print(f'Train size: {len(train_dataset)} \n Test size: {len(test_dataset)}')
    print('==' * 50)
    # fhr_hie_dataset = FHRDataset(hie_list[0:10])
    # hie_dataloader = DataLoader(fhr_hie_dataset, batch_size=1, shuffle=False)

    # define model and train it ----------------------------------------------------------------------------------------
    input_size = config['model_config']['VAE_model']['input_size']
    input_dim = config['model_config']['VAE_model']['input_dim']
    latent_size = config['model_config']['VAE_model']['latent_size']
    latent_dim = config['model_config']['VAE_model']['latent_dim']
    num_LSTM_layers = config['model_config']['VAE_model']['num_LSTM_layers']
    dec_hidden_dim = config['model_config']['VAE_model']['decoder_hidden_dim']
    enc_hidden_dim = config['model_config']['VAE_model']['encoder_hidden_dim']
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
    # VAE_model = VAE_linear(input_seq_size=300, latent_dim=120)
    VAE_model = VAE(input_size=input_size, input_dim=input_dim, latent_size=latent_size, enc_hidden_dim=enc_hidden_dim,
                    latent_dim=latent_dim, num_LSTM_layers=num_LSTM_layers, dec_hidden_dim=dec_hidden_dim,
                    log_stat=log_stat, device=device)
    print(f'Model:  \n {VAE_model}')
    print('==' * 50)
    VAE_model.to(device)
    params = VAE_model.parameters()
    trainable_params = sum(p.numel() for p in VAE_model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable_params}')
    print('==' * 50)
    optimizer = torch.optim.Adam(VAE_model.parameters(), lr=lr)
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[100, 2000])
    epochs = tqdm(range(epochs_num))

    train_loss_list = []
    train_rec_loss_list = []
    train_kld_loss_list = []
    train_nll_loss_list = []

    test_loss_list = []
    test_rec_loss_list = []
    test_kld_loss_list = []
    test_nll_loss_list = []
    vae_loss_re = VAELoss()

    for epoch in tqdm(range(1, epochs_num + 1), desc='Epoch:'):
        log_resource_usage()
        loss_fn = VAELoss()
        loss_train = train(
            model=VAE_model,
            optimizer=optimizer,
            loss_fn=vae_loss_re,
            train_dataloader=train_loader,
            train_plot_dir=train_results_dir,
            epoch_num=epoch
        )




    vae_loss_re = VAELoss()
    train_loss_list = []
    test_loss_list = []
    for epoch in epochs:
        if epoch > 1500:
            update_learning_rate(optimizer, 0.0001)
        else:
            update_learning_rate(optimizer, 0.001)
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
            validation_plot_dir=test_results_dir,
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
