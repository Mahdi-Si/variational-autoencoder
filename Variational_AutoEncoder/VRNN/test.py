import math
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from model import VRNN
import os
import yaml
import logging
from datetime import datetime
import sys
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch.nn.functional as F
import numpy as np
from Variational_AutoEncoder.datasets.custom_datasets import JsonDatasetPreload
from Variational_AutoEncoder.utils.data_utils import plot_scattering, plot_original_reconstructed, \
    calculate_stats, plot_scattering_v2, plot_loss_dict, plot_averaged_results
from Variational_AutoEncoder.utils.run_utils import log_resource_usage
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # remove this line when creating a new environment

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')


if __name__ == '__main__':
    config_file_path = r'config_arguments.yaml'
    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    now = datetime.now()
    run_date = now.strftime("%Y-%m-%d--[%H-%M]-")
    experiment_tag = config['general_config']['tag']
    output_base_dir = os.path.normpath(config['folders_config']['out_dir_base'])
    base_folder = f'{run_date}-{experiment_tag}'
    train_results_dir = os.path.join(output_base_dir, base_folder, 'train_results')
    test_results_dir = os.path.join(output_base_dir, base_folder, 'test_results')
    model_checkpoint_dir = os.path.join(output_base_dir, base_folder, 'model_checkpoints')
    aux_results_dir = os.path.join(output_base_dir, base_folder, 'aux_test_results')
    inference_results_dir = os.path.join(output_base_dir, base_folder, 'inference_results')
    folders_list = [inference_results_dir]
    for folder in folders_list:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # setting up the logging -------------------------------------------------------------------------------------------
    # log_file = os.path.join(train_results_dir, 'log.txt')
    # logging.basicConfig(filename=log_file,
    #                     filemode='w',
    #                     format='%(asctime)s - %(levelname)s - %(message)s',
    #                     level=logging.INFO)
    #
    # sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    #
    # print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    # print('==' * 50)
    # Preparing training and testing datasets --------------------------------------------------------------------------
    dataset_dir = os.path.normpath(config['dataset_config']['dataset_dir'])
    aux_dataset_hie_dir = os.path.normpath(config['dataset_config']['aux_dataset_dir'])
    stat_path = os.path.normpath(config['dataset_config']['stat_path'])
    # batch_size = config['general_config']['batch_size']['train']
    batch_size = 1024
    fhr_healthy_dataset = JsonDatasetPreload(dataset_dir)
    fhr_aux_hie_dataset = JsonDatasetPreload(aux_dataset_hie_dir)
    data_loader_healthy = DataLoader(fhr_healthy_dataset, batch_size=batch_size, shuffle=False)
    data_loader_hie = DataLoader(fhr_aux_hie_dataset, batch_size=batch_size, shuffle=False)

    with open(stat_path, 'rb') as f:
        x_mean = np.load(f)
        x_std = np.load(f)
    log_stat = (x_mean, x_std)
    # fhr_healthy_dataset = FHRDataset(healthy_list)
    torch.manual_seed(42)
    np.random.seed(42)
    # define model and train it ----------------------------------------------------------------------------------------
    raw_input_size = config['model_config']['VAE_model']['raw_input_size']
    input_size = config['model_config']['VAE_model']['input_size']
    input_dim = config['model_config']['VAE_model']['input_dim']
    latent_dim = config['model_config']['VAE_model']['latent_size']
    num_LSTM_layers = config['model_config']['VAE_model']['num_RNN_layers']
    rnn_hidden_dim = config['model_config']['VAE_model']['RNN_hidden_dim']
    epochs_num = config['general_config']['epochs']
    lr = config['general_config']['lr']

    # hyperparameters
    x_dim = input_dim
    h_dim = rnn_hidden_dim
    z_dim = latent_dim
    n_layers = 1
    n_epochs = epochs_num
    clip = 10
    learning_rate = lr
    batch_size = batch_size  # 128
    # seed = 142
    print_every = 20  # batches
    save_every = 20  # epochs

    # manual seed
    # torch.manual_seed(seed)
    plt.ion()

    model = VRNN(x_len=raw_input_size, x_dim=x_dim, h_dim=h_dim, z_dim=z_dim, n_layers=n_layers, log_stat=log_stat)
    params = model.parameters()

    check_point_path = os.path.normpath(r"C:\Users\mahdi\Desktop\Mahdi-Si-Projects\AI\runs\variational-autoencoder\VM\100\VRNN-100.pth")
    checkpoint = torch.load(check_point_path, map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint)
    print(checkpoint.keys())
    # model.load_state_dict(checkpoint['state_dict'])

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, batched_data in tqdm(enumerate(data_loader_healthy), total=len(data_loader_healthy)):
            batched_data = batched_data.to(device)   # (batch_size, signal_len)
            # rec_loss, kld_loss, nll_loss, _, (dec_mean, dec_std), (Sx, meta), z_latent = model(batched_data)
            results = model(batched_data)
            z_latent_ = torch.stack(results.z_latent, dim=2)  # (batch_size, latent_dim, 150)
            h_hidden_ = torch.stack(results.hidden_states, dim=2)
            dec_mean_ = torch.stack(results.decoder_mean, dim=2)  # (batch_size, input_dim, 150)
            Sx = results.Sx.permute(1, 2, 0)  # (batch_size, input_dim, 150)
            selected_idx = np.random.randint(0, batched_data.shape[0], 5)
            for idx in selected_idx:
                selected_signal = batched_data[idx]
                # selected_signal = batched_data[idx].detach().cpu().numpy()
                repeated_signal = selected_signal.repeat(100, 1)
                results = model(repeated_signal)
                # new_sample = model.sample(150).permute(1, 0)
                Sx = results.Sx.permute(1, 2, 0)[0]
                z_latent_ = torch.stack(results.z_latent, dim=2)
                kld_values = torch.stack(results.kld_values, dim=2)
                h_hidden_ = torch.stack(results.hidden_states, dim=2).squeeze(0)
                z_latent_mean = z_latent_.mean(dim=0)
                z_latent_std = z_latent_.std(dim=0)
                h_hidden_mean = h_hidden_.mean(dim=0).permute(1, 0)
                h_hidden_std = h_hidden_.std(dim=0).permute(1, 0)
                dec_mean_ = torch.stack(results.decoder_mean, dim=2)
                dec_mean_mean = dec_mean_.mean(dim=0)
                dec_mean_std = dec_mean_.std(dim=0)
                kld_values_mean = kld_values.mean(dim=0)
                kld_values_mean[:, 0:7] = kld_values_mean[:, 7:8].expand(-1, 7)
                plot_averaged_results(signal=selected_signal.detach().cpu().numpy(), Sx=Sx.detach().cpu().numpy(),
                                      Sxr_mean=dec_mean_mean.detach().cpu().numpy(),
                                      Sxr_std=dec_mean_std.detach().cpu().numpy(),
                                      z_latent_mean=z_latent_mean.detach().cpu().numpy(),
                                      z_latent_std=z_latent_std.detach().cpu().numpy(),
                                      kld_values=kld_values_mean.detach().cpu().numpy(),
                                      h_hidden_mean=h_hidden_mean.detach().cpu().numpy(),
                                      new_sample=new_sample.detach().cpu().numpy(),
                                      plot_dir=inference_results_dir, tag=f'infer_test_{i}_{idx}_average')
                plot_scattering_v2(signal=selected_signal.detach().cpu().numpy(),
                                   Sx=Sx.detach().cpu().numpy(), meta=None,
                                   Sxr=dec_mean_mean.detach().cpu().numpy(),
                                   Sxr_std=dec_mean_std.detach().cpu().numpy(),
                                   z_latent=z_latent_mean.detach().cpu().numpy(),
                                   plot_dir=inference_results_dir, tag=f'infer_test_{i}_{idx}')

                # fig, ax = plt.subplots(nrows=z_latent_selected.shape[0], ncols=1, figsize=(20, 36))
                # i_row = 0
                # for j in range(z_latent_selected.shape[0]):
                #     ax[i_row].plot(z_latent_selected[j, :], linewidth=1.5)
                #     ax[i_row].autoscale(enable=True, axis='x', tight=True)
                #     ax[i_row].set_xticklabels([])
                #     ax[i_row].set_ylabel(f'latent_dim_{j}')
                #     i_row += 1
                #
                # plt.savefig(inference_results_dir + '/' + f'inference_{i}_{idx}' + '_' + '.png', bbox_inches='tight', orientation='landscape', dpi=100)
                # plt.close(fig)
