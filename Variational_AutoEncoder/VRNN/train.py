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
    calculate_stats, plot_scattering_v2, plot_loss_dict
from Variational_AutoEncoder.utils.run_utils import log_resource_usage
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

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
# ----------------------------------------------------------------------------------------------------------------------


def train(epoch=None, model=None, plot_dir=None, tag='', train_loader=None, optimizer=None):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    for param_group in optimizer.param_groups:
        current_learning_rate = param_group['lr']
        print(f'Learning Rate; {current_learning_rate}')
    train_loss = 0
    reconstrucion_loss = 0
    plt.close('all')
    train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    model.train()
    for batch_idx, data in train_loader_tqdm:
        data = data.to(device)
        optimizer.zero_grad()
        rec_loss, kld_loss, nll_loss, _, (dec_mean, dec_std), (Sx, meta), z_latent = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        # grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        z_latent = torch.stack(z_latent, dim=2)
        if batch_idx % print_every == 0:
            message = f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} \
            ({100. * batch_idx / len(train_loader):.0f}%)]\tKLD Loss: {kld_loss.item() / len(data):.6f}\tNLL Loss: \
            {nll_loss.item() / len(data):.6f}'
            tqdm.write(message)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * batch_size, batch_size * (len(train_loader.dataset)//batch_size),
                100. * batch_idx / len(train_loader),
                kld_loss / batch_size,
                nll_loss / batch_size))
            one_data = data[10].unsqueeze(0)
            rec_loss_one, kld_loss_one, nll_loss_one, _, (dec_mean, dec_std), (Sx, meta), z_latent = model(one_data)
            z_latent = torch.stack(z_latent, dim=2)
            sample = model.sample(torch.tensor(150, device=device))
            dec_mean_tensor = torch.cat(dec_mean, dim=0).squeeze()
            dec_std_tensor = torch.cat(dec_std, dim=0).squeeze()
            dec_mean_np = dec_mean_tensor.permute(1, 0).cpu().detach().numpy()
            dec_std_np = dec_std_tensor.cpu().detach().numpy()
            dec_variance_np = np.square(dec_std_np)
            # plot_scattering(signal=data[0], plot_order=[0, 1, (0, 1)], Sx=Sx[0], meta=meta, Sxr=dec_mean_np,
            #                 plot_dir=plot_dir, tag=f'_epoch{epoch}_batch_{batch_idx}_train')
            plot_scattering_v2(signal=one_data.permute(1, 0).detach().cpu().numpy(),
                               Sx=Sx.squeeze(1).permute(1, 0).detach().cpu().numpy(),
                               meta=meta, Sxr=dec_mean_np, z_latent=z_latent.squeeze(0).detach().cpu().numpy(),
                               plot_dir=plot_dir, tag=f'_epoch{epoch}_batch_{batch_idx}_train')
        train_loss += loss.item()
        reconstrucion_loss += rec_loss.item()
    print(f'train loss is ====> {train_loss}')
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    reconstrucion_loss /= len(train_loader.dataset)
    return train_loss, reconstrucion_loss
    

def test(epoch=None, model=None, plot_dir=None, test_loader=None):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    mean_kld_loss, mean_nll_loss, mean_rec_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):

            data = data.to(device)

            rec_loss, kld_loss, nll_loss, _, (dec_mean, dec_std), (Sx, meta), z_latent = model(data)
            mean_kld_loss += kld_loss.item()
            mean_nll_loss += nll_loss.item()
            mean_rec_loss += rec_loss.item()

            one_data = data[0].unsqueeze(0)
            rec_loss_one, kld_loss_one, nll_loss_one, _, (dec_mean, dec_std), (Sx, meta), z_latent_ = model(one_data)
            z_latent = torch.stack(z_latent_, dim=2)
            sample = model.sample(torch.tensor(150, device=device))
            dec_mean_tensor = torch.cat(dec_mean, dim=0).squeeze()  # Remove the unnecessary dimensions
            dec_std_tensor = torch.cat(dec_std, dim=0).squeeze()
            dec_mean_np = dec_mean_tensor.permute(1, 0).cpu().detach().numpy()
            dec_std_np = dec_std_tensor.cpu().detach().numpy()
            dec_variance_np = np.square(dec_std_np)
            # plot_scattering(signal=data[0], plot_order=[0, 1, (0, 1)], Sx=Sx[0], meta=meta, Sxr=dec_mean_np,
            #                 plot_dir=plot_dir, tag=f'_epoch{epoch}_batch_{i}_test')
            plot_scattering_v2(signal=one_data.permute(1, 0).detach().cpu().numpy(),
                               Sx=Sx.squeeze(1).permute(1, 0).detach().cpu().numpy(),
                               meta=meta, Sxr=dec_mean_np, z_latent=z_latent.squeeze(0).detach().cpu().numpy(),
                               plot_dir=plot_dir, tag=f'_epoch{epoch}_batch_{i}_test')

    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)
    mean_rec_loss /= len(test_loader.dataset)
   
    print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
        mean_kld_loss, mean_nll_loss))
    return mean_kld_loss + mean_nll_loss, mean_rec_loss


def aux_hie_test(model=None, dataloader=None, results_dir=None):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    mean_kld_loss, mean_nll_loss, mean_rec_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(device)
            rec_loss, kld_loss, nll_loss, _, (dec_mean, dec_std), (Sx, meta), z_latent = model(data)
            mean_kld_loss += kld_loss.item()
            mean_nll_loss += nll_loss.item()
            mean_rec_loss += rec_loss.item()   # dec_mean -> list 150 of (256, 13) tensor, Sx(150, 256, 13)
            z_latent_ = torch.stack(z_latent, dim=2)  # (256, 9, 150)
            dec_mean_ = torch.stack(dec_mean, dim=2)
            Sx = Sx.permute(1, 2, 0)  # (256, 13, 150)
            selected_idx = np.random.randint(0, data.shape[0], 15)
            for idx in selected_idx:
                selected_signal = data[idx].detach().cpu().numpy()
                Sx_selected = Sx[idx].detach().cpu().numpy()
                dec_mean_selected = dec_mean_[idx].detach().cpu().numpy()
                z_latent_selected = z_latent_[idx].detach().cpu().numpy()
                plot_scattering_v2(signal=selected_signal, Sx=Sx_selected, meta=meta, Sxr=dec_mean_selected,
                                   z_latent=z_latent_selected, plot_dir=results_dir, tag=f'Aux_test_{i}_{idx}')


if __name__ == '__main__':
    config_file_path = r'config_arguments.yaml'
    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    now = datetime.now()
    run_date = now.strftime("%Y-%m-%d--[%H-%M]-")
    experiment_tag = config['general_config']['tag']
    output_base_dir = os.path.normpath(config['folders_config']['out_dir_base'])
    base_folder = f'{run_date}-{experiment_tag}'
    train_results_dir = os.path.join(output_base_dir, base_folder, 'train_results')
    test_results_dir = os.path.join(output_base_dir, base_folder, 'test_results')
    model_checkpoint_dir = os.path.join(output_base_dir, base_folder, 'model_checkpoints')
    aux_results_dir = os.path.join(output_base_dir, base_folder, 'aux_test_results')
    folders_list = [output_base_dir, train_results_dir, test_results_dir, model_checkpoint_dir, aux_results_dir]
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

    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    print('==' * 50)
    # Preparing training and testing datasets --------------------------------------------------------------------------
    dataset_dir = os.path.normpath(config['dataset_config']['dataset_dir'])
    aux_dataset_hie_dir = os.path.normpath(config['dataset_config']['aux_dataset_dir'])
    stat_path = os.path.normpath(config['dataset_config']['stat_path'])
    batch_size = config['general_config']['batch_size']['train']

    # healthy_dataset_path = os.path.join(dataset_dir, 'HEALTHY_signal_dicts.pkl')
    # hie_dataset_path = os.path.join(dataset_dir, 'HIE_signal_dicts.pkl')

    # healthy_list = prepare_data(healthy_dataset_path, do_decimate=False)
    # hie_list = prepare_data(hie_dataset_path, do_decimate=False)

    # fhr_values = [dict_item['fhr'] for dict_item in healthy_list + hie_list]
    # min_fhr = min([min(fhr) for fhr in fhr_values])
    # max_fhr = max([max(fhr) for fhr in fhr_values])
    # normalize_data(healthy_list, min_fhr, max_fhr)
    # normalize_data(hie_list, min_fhr, max_fhr)
    fhr_healthy_dataset = JsonDatasetPreload(dataset_dir)
    fhr_aux_hie_dataset = JsonDatasetPreload(aux_dataset_hie_dir)
    data_loader_complete = DataLoader(fhr_healthy_dataset, batch_size=batch_size, shuffle=False)

    with open(stat_path, 'rb') as f:
        x_mean = np.load(f)
        x_std = np.load(f)

    log_stat = (x_mean, x_std)
    # fhr_healthy_dataset = FHRDataset(healthy_list)
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    aux_hie_loader = DataLoader(fhr_aux_hie_dataset, batch_size=256, shuffle=False, num_workers=4)
    print(f'Train size: {len(train_dataset)} \n Test size: {len(test_dataset)}')
    print('==' * 50)
    # fhr_hie_dataset = FHRDataset(hie_list[0:10])
    # hie_dataloader = DataLoader(fhr_hie_dataset, batch_size=1, shuffle=False)

    # define model and train it ----------------------------------------------------------------------------------------
    input_size = config['model_config']['VAE_model']['input_size']
    input_dim = config['model_config']['VAE_model']['input_dim']
    latent_size = config['model_config']['VAE_model']['latent_size']
    num_LSTM_layers = config['model_config']['VAE_model']['num_LSTM_layers']
    dec_hidden_dim = config['model_config']['VAE_model']['decoder_hidden_dim']
    enc_hidden_dim = config['model_config']['VAE_model']['encoder_hidden_dim']
    epochs_num = config['general_config']['epochs']
    lr = config['general_config']['lr']

    # changing device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    # hyperparameters
    x_dim = input_dim
    h_dim = 90
    z_dim = latent_size
    n_layers = 1
    n_epochs = epochs_num
    clip = 10
    learning_rate = lr
    batch_size = batch_size  # 128
    seed = 142
    print_every = 20  # batches
    save_every = 20  # epochs

    # manual seed
    torch.manual_seed(seed)
    plt.ion()

    model = VRNN(x_dim, h_dim, z_dim, n_layers, log_stat=log_stat)
    print(f'Model:  \n {model}')
    print('==' * 50)
    model = model.to(device)
    params = model.parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable_params}')
    print('==' * 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[100, 2000])
    train_loss_list = []
    train_rec_loss_list = []
    test_rec_loss_list = []
    test_loss_list = []
    for epoch in tqdm(range(1, n_epochs + 1), desc='Epoch:'):
        log_resource_usage()
        train_loss, train_rec_loss = train(model=model, epoch=epoch, plot_dir=train_results_dir,
                                           train_loader=train_loader, optimizer=optimizer)

        if len(train_loss_list) > 0:
            if train_loss <= min(train_loss_list):
                checkpoint_name = f'VRNN-{epoch}.pth'
                model_dir = os.path.join(model_checkpoint_dir, checkpoint_name)
                for file_name in os.listdir(model_checkpoint_dir):
                    if file_name.endswith('.pth'):
                        os.remove(os.path.join(model_checkpoint_dir, file_name))
                torch.save(model.state_dict(), model_dir)

        schedular.step()
        train_loss_list.append(train_loss)
        train_rec_loss_list.append(train_rec_loss)

        test_loss, test_rec_loss = test(epoch=epoch, model=model, plot_dir=test_results_dir, test_loader=test_loader)
        test_loss_list.append(test_loss)
        test_rec_loss_list.append(test_rec_loss)

        tqdm.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

        train_loss_min = min(train_loss_list)

        # if epoch % save_every == 1:
        #     fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
        #     torch.save(model.state_dict(), fn)
        #     print('Saved model to '+fn)

        loss_dict = {'train': train_loss_list,
                     'test': test_loss_list,
                     'train_rec': train_rec_loss_list,
                     'test_rec': test_rec_loss_list}
        loss_path = os.path.join(train_results_dir, 'loss_dict.pkl')
        with open(loss_path, 'wb') as file:
            pickle.dump(loss_dict, file)
        plot_loss_dict(loss_dict=loss_dict, epoch_num=epoch, plot_dir=train_results_dir)

    aux_hie_test(model=model, dataloader=aux_hie_loader, results_dir=aux_results_dir)