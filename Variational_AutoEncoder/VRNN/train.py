
import torch.nn as nn
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
import os
import yaml
import logging
from datetime import datetime
import sys
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from vrnn_classifier_gauss_experiment_1 import VRNNGauss
from Variational_AutoEncoder.datasets.custom_datasets import JsonDatasetPreload, FhrUpPreload
from Variational_AutoEncoder.utils.data_utils import plot_scattering_v2, plot_loss_dict
from Variational_AutoEncoder.utils.run_utils import log_resource_usage, StreamToLogger, setup_logging
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------------------------------------------------------------------------


def train(epoch_train=None, model=None, kld_beta=1.1, plot_dir=None, tag='', train_loader=None,
          optimizer=None, plot_every_epoch=None):
    for param_group in optimizer.param_groups:
        current_learning_rate = param_group['lr']
        print(f'Learning Rate; {current_learning_rate}')
    train_loss_epoch = 0
    reconstruction_loss_epoch = 0
    kld_loss_epoch = 0
    nll_loss_epoch = 0
    plt.close('all')
    train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_train}")
    model.train()
    for batch_idx, train_data in train_loader_tqdm:
        data = train_data[0]
        data = data.to(device)
        optimizer.zero_grad()
        results = model(data)
        loss = (kld_beta * results.kld_loss) + results.nll_loss
        loss.backward()
        optimizer.step()
        kld_loss_epoch += kld_beta * results.kld_loss.item()
        # nll_loss_epoch += results.nll_loss.item()
        train_loss_epoch += loss.item()
        reconstruction_loss_epoch += results.rec_loss.item()

        # grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        z_latent = torch.stack(results.z_latent, dim=2)
        message = (f'Train Epoch: {epoch_train} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                   f'({100. * batch_idx / len(train_loader):.0f}%)] | '
                   f'-KLD Loss: {results.kld_loss.item():.5f} - Weighted KLD Loss: {kld_beta * results.kld_loss:.5f} | '
                   f'-Reconstruction Loss: {results.rec_loss.item():.5f}')
        print(message)
        # tqdm.write(message)
        if epoch_train % plot_every_epoch == 0:
            if batch_idx % 100 == 0:
                signal_ = data[0]
                sx_ = results.Sx.permute(1, 2, 0)[0]
                z_latent_ = torch.stack(results.z_latent, dim=2)[0]
                dec_mean_ = torch.stack(results.decoder_mean, dim=2)[0]
                # signal = signal_.squeeze(0).permute(1, 0).detach().cpu().numpy()  # for two channels
                signal = signal_.detach().cpu().numpy()  # for one channels
                plot_scattering_v2(signal=signal,
                                   Sx=sx_.detach().cpu().numpy(),
                                   meta=None,
                                   plot_second_channel=False,
                                   Sxr=dec_mean_.detach().cpu().numpy(),
                                   z_latent=z_latent_.detach().cpu().numpy(),
                                   plot_dir=plot_dir, tag=f'_epoch{epoch_train}_batch_{batch_idx}_train')

    # print(f'Train Loop train loss is ====> {train_loss_tl}')
    # print('====> Epoch: {} Average loss: {:.4f}'.format(
    #     epoch, train_loss_tl / len(train_loader.dataset)))
    train_loss_tl_avg = train_loss_epoch / len(train_loader.dataset)
    reconstruction_loss_avg = reconstruction_loss_epoch / len(train_loader.dataset)
    kld_loss_avg = kld_loss_epoch / len(train_loader.dataset)

    print(f'Train Loss Mean: {train_loss_tl_avg} - Reconstruction Loss Mean: {reconstruction_loss_avg}')
    return train_loss_tl_avg, reconstruction_loss_avg, kld_loss_avg
    

def test(epoch_test=None, model=None, plot_dir=None, test_loader=None, plot_every_epoch=None, kld_beta=1.1):
    mean_test_loss, mean_kld_loss, mean_nll_loss, mean_rec_loss = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            data = test_data[0]
            data = data.to(device)

            results_test = model(data)
            mean_test_loss += (kld_beta * results_test.kld_loss.item() + results_test.rec_loss.item())
            mean_kld_loss += kld_beta * results_test.kld_loss.item()
            # mean_nll_loss += results_test.nll_loss.item()
            mean_rec_loss += results_test.rec_loss.item()

            one_data = data[0].unsqueeze(0)
            results_test_ = model(one_data)
            z_latent = torch.stack(results_test_.z_latent, dim=2)
            # sample = model.sample(torch.tensor(150, device=device))
            dec_mean_tensor = torch.cat(results_test_.decoder_mean, dim=0)  # Remove the unnecessary dimensions
            dec_std_tensor = torch.cat(results_test_.decoder_std, dim=0)
            dec_mean_np = dec_mean_tensor.permute(1, 0).cpu().detach().numpy()
            dec_std_np = dec_std_tensor.cpu().detach().numpy()
            dec_variance_np = np.square(dec_std_np)
            if epoch_test % plot_every_epoch == 0:
            # if epoch > 0:
            #     signal = one_data.squeeze(0).permute(1, 0).detach().cpu().numpy()   # for two channels
                signal = one_data.detach().cpu().numpy()  # for one channels
                plot_scattering_v2(signal=signal, plot_second_channel=False,
                                   Sx=results_test_.Sx.squeeze(1).permute(1, 0).detach().cpu().numpy(),
                                   meta=None, Sxr=dec_mean_np, z_latent=z_latent.squeeze(0).detach().cpu().numpy(),
                                   plot_dir=plot_dir, tag=f'_epoch{epoch_test}_batch_{i}_test')

    mean_test_loss /= len(test_loader.dataset)
    mean_kld_loss /= len(test_loader.dataset)
    # mean_nll_loss /= len(test_loader.dataset)
    mean_rec_loss /= len(test_loader.dataset)
   
    # print(f'Average Test set loss: KLD Loss = {mean_kld_loss}, \
    #  NLL Loss = {mean_nll_loss}, \
    #   reconstruction loss = {mean_rec_loss}')
    return mean_test_loss, mean_rec_loss, mean_kld_loss


def aux_hie_test(model=None, dataloader=None, results_dir=None):
    mean_kld_loss, mean_nll_loss, mean_rec_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(device)
            results_aux = model(data)
            mean_kld_loss += results_aux.kld_loss.item()
            # mean_nll_loss += results_aux.nll_loss.item()
            mean_rec_loss += results_aux.rec_loss.item()   # dec_mean -> list 150 of (256, 13) tensor, Sx(150, 256, 13)
            z_latent_ = torch.stack(results_aux.z_latent, dim=2)  # (256, 9, 150)
            dec_mean_ = torch.stack(results_aux.decoder_mean, dim=2)
            Sx = results_aux.Sx.permute(1, 2, 0)  # (256, 13, 150)
            selected_idx = np.random.randint(0, data.shape[0], 15)
            for idx in selected_idx:
                selected_signal = data[idx].detach().cpu().numpy()
                Sx_selected = Sx[idx].detach().cpu().numpy()
                dec_mean_selected = dec_mean_[idx].detach().cpu().numpy()
                z_latent_selected = z_latent_[idx].detach().cpu().numpy()
                plot_scattering_v2(signal=selected_signal, Sx=Sx_selected, meta=None, Sxr=dec_mean_selected,
                                   z_latent=z_latent_selected, plot_dir=results_dir, tag=f'Aux_test_{i}_{idx}')


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
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
    aux_dir = os.path.join(output_base_dir, base_folder, 'aux_test_HIE')
    tensorboard_dir = os.path.join(output_base_dir, base_folder, 'tensorboard_log')
    folders_list = [output_base_dir, train_results_dir, test_results_dir, model_checkpoint_dir, aux_dir,
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
    lr_milestones = config['general_config']['lr_milestone']
    dataset_dir = os.path.normpath(config['dataset_config']['dataset_dir'])
    aux_dataset_hie_dir = os.path.normpath(config['dataset_config']['aux_dataset_dir'])
    stat_path = os.path.normpath(config['dataset_config']['stat_path'])
    batch_size = config['general_config']['batch_size']['train']
    plot_every_epoch = config['general_config']['plot_frequency']
    previous_check_point = config['general_config']['checkpoint_path']

    # define model and train it ----------------------------------------------------------------------------------------
    raw_input_size = config['model_config']['VAE_model']['raw_input_size']
    input_size = config['model_config']['VAE_model']['input_size']
    input_dim = config['model_config']['VAE_model']['input_dim']
    latent_dim = config['model_config']['VAE_model']['latent_size']
    num_layers = config['model_config']['VAE_model']['num_RNN_layers']
    rnn_hidden_dim = config['model_config']['VAE_model']['RNN_hidden_dim']
    epochs_num = config['general_config']['epochs']
    lr = config['general_config']['lr']
    kld_beta_ = config['model_config']['VAE_model']['kld_beta']
    kld_beta_ = float(kld_beta_)

    # hyperparameters
    x_dim = input_dim
    h_dim = rnn_hidden_dim
    z_dim = latent_dim
    n_layers = num_layers
    n_epochs = epochs_num
    clip = 10
    learning_rate = lr
    batch_size = batch_size
    plt.ion()

    # dataset creation -------------------------------------------------------------------------------------------------
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
    # fhr_healthy_dataset = FhrUpPreload(dataset_dir)
    # fhr_aux_hie_dataset = JsonDatasetPreload(aux_dataset_hie_dir)
    # data_loader_complete = DataLoader(fhr_healthy_dataset, batch_size=batch_size, shuffle=False)

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    # aux_hie_loader = DataLoader(fhr_aux_hie_dataset, batch_size=256, shuffle=False, num_workers=32)
    print(f'Train size: {len(train_dataset)} \n Test size: {len(test_dataset)}')
    print('==' * 50)
    # fhr_hie_dataset = FHRDataset(hie_list[0:10])
    # hie_dataloader = DataLoader(fhr_hie_dataset, batch_size=1, shuffle=False)



    # model = VRNN(x_len=raw_input_size, x_dim=x_dim, h_dim=h_dim, z_dim=z_dim, n_layers=n_layers, log_stat=log_stat)
    model = VRNNGauss(input_dim=input_dim, input_size=raw_input_size, h_dim=h_dim, z_dim=z_dim,
                      n_layers=n_layers, device=device, log_stat=log_stat, bias=False)

    # model = VRNN_GMM(input_dim=input_dim, input_size=raw_input_size, h_dim=h_dim, z_dim=z_dim,
    #                  n_layers=n_layers, device=device, log_stat=log_stat, bias=False)

    print(f'Model:  \n {model}')
    print('==' * 50)
    model = model.to(device)
    params = model.parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable_params}')
    print('==' * 50)
    # writer = SummaryWriter(log_dir=tensorboard_dir)
    # writer.add_graph(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_milestones)

    if previous_check_point is not None:
        print(f"Loading checkpoint '{previous_check_point}'")
        checkpoint = torch.load(previous_check_point)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{previous_check_point}' (epoch {checkpoint['epoch']})")
    else:
        start_epoch = 1
    train_loss_list = []
    train_rec_loss_list = []
    train_kld_loss_list = []
    # train_nll_loss_list = []

    test_loss_list = []
    test_rec_loss_list = []
    test_kld_loss_list = []
    # test_nll_loss_list = []
    for epoch in tqdm(range(start_epoch, n_epochs + 1), desc='Epoch:'):
        # log_resource_usage()
        # if epoch == 2000:
        #     new_batch_size = batch_size // 2  # Reduce batch size
        #     print(f'Reducing batch size to {new_batch_size}')
        #     train_loader = DataLoader(train_dataset, batch_size=new_batch_size, shuffle=True, num_workers=20)
        #     test_loader = DataLoader(test_dataset, batch_size=new_batch_size, shuffle=False, num_workers=20)
        train_loss, train_rec_loss, train_kld_loss = train(model=model, epoch_train=epoch,
                                                           plot_dir=train_results_dir,
                                                           plot_every_epoch=plot_every_epoch,
                                                           train_loader=train_loader,
                                                           kld_beta=kld_beta_,
                                                           optimizer=optimizer)

        train_loss_list.append(train_loss)
        train_rec_loss_list.append(train_rec_loss)
        train_kld_loss_list.append(train_kld_loss)
        # train_nll_loss_list.append(train_nll_loss)

        if len(train_loss_list) > 0:
            if train_loss <= min(train_loss_list):
                checkpoint_name = f'VRNN-{epoch}.pth'
                model_dir = os.path.join(model_checkpoint_dir, checkpoint_name)
                # for file_name in os.listdir(model_checkpoint_dir):
                #     if file_name.endswith('.pth'):
                #         os.remove(os.path.join(model_checkpoint_dir, file_name))

                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, model_dir)

        schedular.step()
        test_loss, test_rec_loss, test_kld_loss = test(epoch_test=epoch, model=model, plot_dir=test_results_dir,
                                                       test_loader=test_loader, plot_every_epoch=plot_every_epoch,
                                                       kld_beta=kld_beta_)
        test_loss_list.append(test_loss)
        test_rec_loss_list.append(test_rec_loss)
        test_kld_loss_list.append(test_kld_loss)

        tqdm.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

        train_loss_min = min(train_loss_list)

        loss_dict = {'train_loss': train_loss_list,
                     'test_loss': test_loss_list,
                     'train_rec_loss': train_rec_loss_list,
                     'test_rec_loss': test_rec_loss_list,
                     'train_kld_loss': train_kld_loss_list,
                     'test_kld_loss': test_kld_loss_list,
                     }

        # writer.add_scalar('Train/Loss', train_loss, epoch)
        # writer.add_scalar('Test/Loss', test_loss, epoch)
        # writer.add_scalar('Train/Reconstruction_Loss', train_rec_loss, epoch)
        # writer.add_scalar('Test/Reconstruction_Loss', test_rec_loss, epoch)
        loss_path = os.path.join(train_results_dir, 'loss_dict.pkl')
        if epoch % plot_every_epoch == 0:
            with open(loss_path, 'wb') as file:
                pickle.dump(loss_dict, file)
            plot_loss_dict(loss_dict=loss_dict, epoch_num=epoch, plot_dir=train_results_dir)
        # writer.flush()
        # writer.close()

    # aux_hie_test(model=model, dataloader=aux_hie_loader, results_dir=aux_results_dir)
