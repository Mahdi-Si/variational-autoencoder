import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from tqdm import tqdm
import psutil
import GPUtil

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
import torch.nn as nn
import numpy as np

from Variational_AutoEncoder.datasets.custom_datasets import JsonDatasetPreload
from Variational_AutoEncoder.utils.data_utils import plot_scattering, plot_original_reconstructed, \
    calculate_stats
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
    train_loss = 0
    plt.close('all')
    train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    # for batch_idx, (data) in enumerate(train_loader):
    model.train()
    for batch_idx, data in train_loader_tqdm:
        #transforming data
        data = data.to(device)
        # data = data.squeeze().transpose(0, 1)  # (seq, batch, elem)
        # data = (data - data.min()) / (data.max() - data.min())
        
        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, (dec_mean, dec_std), (Sx, meta) = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        # grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        if batch_idx % print_every == 0:
            message = f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tKLD Loss: {kld_loss.item() / len(data):.6f}\tNLL Loss: {nll_loss.item() / len(data):.6f}'
            tqdm.write(message)
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
            #     epoch, batch_idx * batch_size, batch_size * (len(train_loader.dataset)//batch_size),
            #     100. * batch_idx / len(train_loader),
            #     kld_loss / batch_size,
            #     nll_loss / batch_size))
            one_data = data[10].unsqueeze(0)
            kld_loss_one, nll_loss_one, _, (dec_mean, dec_std), (Sx, meta) = model(one_data)
            sample = model.sample(torch.tensor(150, device=device))
            # plt.imshow(sample.to(torch.device('cpu')).numpy())

            # this part is for only order 0 scattering transform and the plots -----------------------------------------
            # plot_original_reconstructed(scattering_original_one.squeeze(2).detach().cpu().numpy(),
            #                             sample.detach().cpu().numpy(),
            #                             plot_dir=plot_dir,
            #                             tag=f'train_{str(epoch)}_{batch_idx}')
            # dec_mean_tensor = torch.cat(dec_mean, dim=0).squeeze()  # Remove the unnecessary dimensions
            # dec_std_tensor = torch.cat(dec_std, dim=0).squeeze()
            # dec_mean_np = dec_mean_tensor.cpu().detach().numpy()
            # dec_std_np = dec_std_tensor.cpu().detach().numpy()
            # dec_variance_np = np.square(dec_std_np)
            # time_vector = np.arange(len(dec_mean_np))
            # plt.figure(figsize=(18, 7))
            # plt.plot(time_vector, dec_mean_np, label='Mean')
            # plt.plot(time_vector, scattering_original_one.squeeze(2).detach().cpu().numpy(), label='Original Signal')
            # plt.fill_between(time_vector,
            #                  dec_mean_np - dec_variance_np,  # Lower bound
            #                  dec_mean_np + dec_variance_np,  # Upper bound
            #                  color='blue', alpha=0.1, label='Variance')
            # plt.xlabel('Time')
            # plt.ylabel('Signal')
            # plt.title('Signal with Probability (Mean and Variance)')
            # plt.legend()
            # tag = f'train_{str(epoch)}_{batch_idx}_prob'
            # plt.savefig(plot_dir + '/' + tag + '_' + '_st.png', bbox_inches='tight', orientation='landscape')
            # plt.close()
            # this part is for only order 0 scattering transform and the plots -----------------------------------------

            dec_mean_tensor = torch.cat(dec_mean, dim=0).squeeze()  # Remove the unnecessary dimensions
            dec_std_tensor = torch.cat(dec_std, dim=0).squeeze()
            dec_mean_np = dec_mean_tensor.cpu().detach().numpy()
            dec_std_np = dec_std_tensor.cpu().detach().numpy()
            dec_variance_np = np.square(dec_std_np)
            plot_scattering(signal=data[0], plot_order=[0, 1, (0, 1)], Sx=Sx[0], meta=meta, Sxr=dec_mean_np,
                            plot_dir=plot_dir, tag=f'_epoch{epoch}_batch_{batch_idx}_train')
        train_loss += loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    return train_loss
    

def test(epoch=None, model=None, plot_dir=None, test_loader=None):
    """uses test data to evaluate 
    likelihood of the model"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    mean_kld_loss, mean_nll_loss = 0, 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):

            data = data.to(device)
            # data = data.squeeze().transpose(0, 1)
            # data = (data - data.min()) / (data.max() - data.min())

            kld_loss, nll_loss, _, (dec_mean, dec_std), (Sx, meta) = model(data)
            mean_kld_loss += kld_loss.item()
            mean_nll_loss += nll_loss.item()

            one_data = data[0].unsqueeze(0)
            kld_loss_one, nll_loss_one, _, (dec_mean, dec_std), (Sx, meta) = model(one_data)
            sample = model.sample(torch.tensor(150, device=device))
            dec_mean_tensor = torch.cat(dec_mean, dim=0).squeeze()  # Remove the unnecessary dimensions
            dec_std_tensor = torch.cat(dec_std, dim=0).squeeze()
            dec_mean_np = dec_mean_tensor.cpu().detach().numpy()
            dec_std_np = dec_std_tensor.cpu().detach().numpy()
            dec_variance_np = np.square(dec_std_np)
            plot_scattering(signal=data[0], plot_order=[0, 1, (0, 1)], Sx=Sx[0], meta=meta, Sxr=dec_mean_np,
                            plot_dir=plot_dir, tag=f'_epoch{epoch}_batch_{i}_test')

    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)
   
    print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
        mean_kld_loss, mean_nll_loss))
    return mean_kld_loss + mean_nll_loss


if __name__ == '__main__':
    config_file_path = r'config_arguments.yaml'
    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    print('==' * 50)

    now = datetime.now()
    run_date = now.strftime("%Y-%m-%d---[%H-%M]")
    experiment_tag = config['general_config']['tag']
    output_base_dir = os.path.normpath(config['folders_config']['out_dir_base'])
    base_folder = f'{run_date}-{experiment_tag}'
    train_results_dir = os.path.join(output_base_dir, base_folder, 'train_results')
    test_results_dir = os.path.join(output_base_dir, base_folder, 'test_results')
    model_checkpoint_dir = os.path.join(output_base_dir, base_folder, 'model_checkpoints')
    folders_list = [output_base_dir, train_results_dir, test_results_dir, model_checkpoint_dir]
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
    dataset_dir = os.path.normpath(config['dataset_config']['dataset_dir'])
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
    data_loader_complete = DataLoader(fhr_healthy_dataset, batch_size=batch_size, shuffle=False)
    mean_, std_ = calculate_stats(data_loader_complete)

    fhr_healthy_dataset_normalized = JsonDatasetPreload(dataset_dir, mean=mean_, std=std_)
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

    train_dataset, test_dataset = random_split(fhr_healthy_dataset_normalized, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
    x_dim = 13
    h_dim = 10
    z_dim = 100
    n_layers = 1
    n_epochs = 2000
    clip = 10
    learning_rate = 1e-3
    batch_size = 64 #128
    seed = 128
    print_every = 20  # batches
    save_every = 20  # epochs

    # manual seed
    torch.manual_seed(seed)
    plt.ion()

    # init model + optimizer + datasets

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=True, download=True,
    #                    transform=transforms.ToTensor()),
    #     batch_size=batch_size, shuffle=True)
    #
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=False,
    #                    transform=transforms.ToTensor()),
    #     batch_size=batch_size, shuffle=True)

    model = VRNN(x_dim, h_dim, z_dim, n_layers)

    print(f'Model:  \n {model}')
    print('==' * 50)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_list = []
    test_loss_list = []
    for epoch in tqdm(range(1, n_epochs + 1), desc='Epoch:'):
        log_resource_usage()
        # training + testing
        train_loss = train(model=model, epoch=epoch, plot_dir=train_results_dir, train_loader=train_loader,
                           optimizer=optimizer)
        train_loss_list.append(train_loss)

        test_loss = test(epoch=epoch, model=model, plot_dir=test_results_dir, test_loader=test_loader)
        test_loss_list.append(test_loss)
        tqdm.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
        # # saving model
        # if epoch % save_every == 1:
        #     fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
        #     torch.save(model.state_dict(), fn)
        #     print('Saved model to '+fn)

    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 32
    plt.rcParams['text.usetex'] = True
    loss_dict = {'train': train_loss_list, 'test': test_loss_list}
    loss_path = os.path.join(train_results_dir, 'loss_dict.pkl')
    with open(loss_path, 'wb') as file:
        pickle.dump(loss_dict, file)
    t = np.arange(1, n_epochs + 1)
    print(f'len t is {len(t)} and len train_loss_list is {len(train_loss_list)}')
    ax[0].autoscale(enable=True, axis='x', tight=True)
    ax[0].plot(train_loss_list, label='train loss', color='#FF5733',
               linewidth=1)  # Custom hex color and line thickness
    ax[1].autoscale(enable=True, axis='x', tight=True)
    ax[1].plot(test_loss_list, label='test loss', color='#005B41',
               linewidth=1)  # Custom hex color and line thickness
    # Adding grid, legend, and labels with specific requirements
    # ax[0].grid(True)
    # ax[0].legend()
    # ax[0].set_xlabel('$\ell_2$')
    plt.savefig(f'{train_results_dir}/Loss_st.pdf', bbox_inches='tight')