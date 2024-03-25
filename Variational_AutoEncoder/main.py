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
from Variational_AutoEncoder.utils.run_utils import log_resource_usage, StreamToLogger, setup_logging

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    kld = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + (0.001 * kld)


# Todo implement log_stat for config and here
def train(model=None, optimizer=None, loss_fn=None, train_dataloader=None, train_plot_dir=None, epoch_num=None,
          plot_every_epoch=None):
    for param_group in optimizer.param_groups:
        current_learning_rate = param_group['lr']
        print(f'Learning Rate; {current_learning_rate}')

    train_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="training", leave=False)
    model.train()
    loss_epoch = 0
    reconstruction_loss_epoch = 0
    kld_loss_epoch = 0
    for batch_idx, batch_data in train_iterator:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        scattering_original, x_rec, z, mean, logvar = model(batch_data)
        # input output signals: (Batch_size, input_dim, input_size)
        # z, mean, logvar: (batch_size, latent_size, latent_dim)
        total_loss, recon_loss, kl_div = loss_fn(x_rec, scattering_original, mean, logvar)
        train_iterator.set_postfix(total_loss=total_loss.item(),
                                   recon_loss=recon_loss.item(),
                                   kl_div=kl_div.item())
        total_loss.backward()
        optimizer.step()

        loss_epoch += total_loss.item()
        reconstruction_loss_epoch += recon_loss.item()
        kld_loss_epoch += kl_div.item()
        nn.utils.clip_grad_norm_(model.parameters(), 12)  # observe the performance with respect to clip
        if epoch % plot_every_epoch == 0 and batch_idx == (len(train_iterator)-1):
            selected_idx = np.random.randint(0, batch_data.shape[0], 1)
            for idx in selected_idx:
                plot_scattering_v2(signal=batch_data[idx].unsqueeze(1).detach().cpu().numpy(),
                                   Sx=scattering_original[idx].detach().cpu().numpy(),
                                   Sxr=x_rec[idx].detach().cpu().numpy(),
                                   z_latent=z[idx].permute(1, 0).detach().cpu().numpy(),
                                   tag=f'train_{epoch_num}_{batch_idx}', plot_dir=train_plot_dir)
                # plot_original_reconstructed(original_x=original_x[idx[0]].cpu().detach().numpy(),
                #                             reconstructed_x=reconstructed_x[idx[0]].cpu().detach().numpy(),
                #                             plot_dir=train_plot_dir, tag=f'train_{epoch_num}_{batch_idx}_')
    avr_total_loss = loss_epoch / len(train_dataloader)
    avr_recon_loss = reconstruction_loss_epoch / len(train_dataloader)
    avr_kld_loss = kld_loss_epoch / len(train_dataloader)
    train_iterator.set_postfix(total_loss=avr_total_loss,
                               recon_loss=avr_recon_loss,
                               kl_div=avr_kld_loss)
    return avr_total_loss, avr_recon_loss, avr_kld_loss


def test(model=None, loss_fn=None, test_dataloader=None, validation_plot_dir=None, epoch_num=None,
         plot_every_epoch=None):
    model.eval()
    test_iterator = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="testing")
    loss_epoch = 0
    reconstruction_loss_epoch = 0
    kld_loss_epoch = 0
    with torch.no_grad():
        for batch_idx, batch_data in test_iterator:
            batch_data = batch_data.to(device)
            scattering_original, x_rec, z, mean, logvar = model(batch_data)
            total_loss, recon_loss, kl_div = loss_fn(scattering_original, x_rec, mean, logvar)
            loss_epoch += total_loss.item()
            reconstruction_loss_epoch += recon_loss.item()
            kld_loss_epoch += kl_div.item()
            if epoch % plot_every_epoch == 0:
                selected_idx = np.random.randint(0, batch_data.shape[0], 3)
                if batch_idx % 100 == 0:
                    # recon_x = reconstructed_x.detach().cpu().numpy()
                    # true_fhr = batch_data.cpu().numpy()  # nd(64, 300)
                    indices = torch.randperm(scattering_original.size(0))[:5]
                    original_x_sampled = scattering_original[indices]
                    reconstructed_x_sampled = scattering_original[indices]
                    selected_idx = np.random.randint(0, len(scattering_original)-1, 5)
                    for i in selected_idx:
                        pass
                        # plot_original_reconstructed(original_x=scattering_original[i].cpu().detach().numpy(),
                        #                             reconstructed_x=scattering_original[i].cpu().detach().numpy(),
                        #                             plot_dir=validation_plot_dir, tag=f'test_{epoch_num}_{i}_')
        avr_total_loss = loss_epoch / len(test_dataloader)
        avr_recon_loss = reconstruction_loss_epoch / len(test_dataloader)
        avr_kld_loss = kld_loss_epoch / len(test_dataloader)
        test_iterator.set_postfix(total_loss=avr_total_loss,
                                  recon_loss=avr_recon_loss,
                                  kl_div=avr_kld_loss)
        return avr_total_loss, avr_recon_loss, avr_kld_loss

if __name__ == "__main__":
    # read config file -------------------------------------------------------------------------------------------------
    config_file_path = r'config_arguments.yaml'
    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    print('==' * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # define model and train it ----------------------------------------------------------------------------------------
    input_size = config['model_config']['VAE_model']['input_size']
    input_dim = config['model_config']['VAE_model']['input_dim']
    latent_size = config['model_config']['VAE_model']['latent_size']
    latent_dim = config['model_config']['VAE_model']['latent_dim']
    num_LSTM_layers = config['model_config']['VAE_model']['num_LSTM_layers']
    dec_hidden_dim = config['model_config']['VAE_model']['decoder_hidden_dim']
    enc_hidden_dim = config['model_config']['VAE_model']['encoder_hidden_dim']
    lr = config['general_config']['lr']
    checkpoint_frequency = config['general_config']['checkpoint_frequency']
    # model ------------------------------------------------------------------------------------------------------------
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
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1000])
    epochs = tqdm(range(epochs_num))

    train_loss_list = []
    train_rec_loss_list = []
    train_kld_loss_list = []
    train_nll_loss_list = []

    test_loss_list = []
    test_rec_loss_list = []
    test_kld_loss_list = []
    test_nll_loss_list = []
    vae_loss_re = VAELoss(beta=5)

    for epoch in tqdm(range(1, epochs_num + 1), desc='Epoch:'):
        log_resource_usage()
        loss_fn = VAELoss()
        train_loss, train_rec_loss, train_kld_loss = train(
            model=VAE_model,
            optimizer=optimizer,
            loss_fn=vae_loss_re,
            train_dataloader=train_loader,
            train_plot_dir=train_results_dir,
            epoch_num=epoch,
            plot_every_epoch=plot_every_epoch
        )

        if len(train_loss_list) > 0 and epoch % checkpoint_frequency == 0:
            if train_loss <= min(train_loss_list):
                checkpoint_name = f'VAE-{epoch}.pth'
                model_dir = os.path.join(model_checkpoint_dir, checkpoint_name)
                # for file_name in os.listdir(model_checkpoint_dir):
                #     if file_name.endswith('.pth'):
                #         os.remove(os.path.join(model_checkpoint_dir, file_name))
                torch.save(VAE_model.state_dict(), model_dir)

        train_loss_list.append(train_loss)
        train_rec_loss_list.append(train_rec_loss)
        train_kld_loss_list.append(train_kld_loss)

        test_loss, test_rec_loss, test_kld_loss = test(
            model=VAE_model,
            loss_fn=vae_loss_re,
            test_dataloader=test_loader,
            validation_plot_dir=test_results_dir,
            epoch_num=epoch,
            plot_every_epoch=plot_every_epoch
        )
        tqdm.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
        test_loss_list.append(test_loss)
        test_rec_loss_list.append(test_rec_loss)
        test_kld_loss_list.append(test_kld_loss)

        loss_dict = {'train_loss': train_loss_list,
                     'test_loss': test_loss_list,
                     'train_rec_loss': train_rec_loss_list,
                     'test_rec_loss': test_rec_loss_list,
                     'train_kld_loss': train_kld_loss_list,
                     'test_kld_loss': test_kld_loss_list}

        loss_path = os.path.join(train_results_dir, 'loss_dict.pkl')
        if epoch % plot_every_epoch == 0:
            with open(loss_path, 'wb') as file:
                pickle.dump(loss_dict, file)
            plot_loss_dict(loss_dict=loss_dict, epoch_num=epoch, plot_dir=train_results_dir)

    loss_path = os.path.join(train_results_dir, 'loss_dict.pkl')
    with open(loss_path, 'wb') as file:
        pickle.dump(loss_dict, file)
    plot_loss_dict(loss_dict=loss_dict, epoch_num=-1, plot_dir=train_results_dir)
