
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
import os
import yaml
import logging
from datetime import datetime
import sys
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.distributions as tdist
import numpy as np
from sklearn.manifold import TSNE
from Variational_AutoEncoder.datasets.custom_datasets import JsonDatasetPreload, RepeatSampleDataset, FhrUpPreload
from Variational_AutoEncoder.utils.data_utils import plot_scattering_v2, plot_averaged_results, plot_general_mse, \
    plot_generated_samples, plot_distributions, plot_histogram
import seaborn as sns
# from vrnn_gauss import VRNN_Gauss
from vrnn_gauss_GMM_experiment_8 import VRNNGauss
from Variational_AutoEncoder.utils.run_utils import log_resource_usage, StreamToLogger, setup_logging
import pandas as pd
import plotly.graph_objects as go

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # remove this line when creating a new environment

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')


# def calculate_log_likelihood_generated(sample, sample_mu, sample_sigma):
#     dist = torch.distributions.Normal(sample_mu, sample_sigma)
#     log_probs = dist.log_prob(sample)
#     total_log_likelihood = log_probs.sum()
#     return total_log_likelihood

def calculate_log_likelihood_generated(samples, mean, sigma):
    dist = torch.distributions.Normal(mean, sigma)
    log_likelihood = dist.log_prob(samples)
    return log_likelihood


def calculate_log_likelihood(dec_mean_t_, dec_std_t_, Sx_t_):
    # Ensure the inputs are in the correct shape and on the correct device
    dec_mean_t_ = dec_mean_t_.to(Sx_t_.device)
    dec_std_t_ = dec_std_t_.to(Sx_t_.device)
    # Create a normal distribution with the predicted mean and std
    pred_dist = tdist.Normal(dec_mean_t_, dec_std_t_)
    # Compute the log probability of the original signal under this distribution
    log_probs = pred_dist.log_prob(Sx_t_)
    # Sum the log probabilities over the input dimensions and sequence length
    log_likelihoods = log_probs.sum(dim=[1, 2])
    return log_likelihoods.cpu().numpy()


def run_test_full_data(model_t, data_loader, input_dim_t, average_results=False, plot_selected=False,
                       modify_h=None, modify_z=None, base_dir=None, channel_num=1, tag='_'):
    model_t.modify_z = modify_z
    model_t.modify_h = modify_h
    model_t.to(device)
    model_t.eval()
    mse_all_data = torch.empty((0, input_dim_t)).to(device)


def plot_tSNE(model_t, data_loader, input_dim_t, average_results=False, plot_selected=False,
              modify_h=None, modify_z=None, base_dir=None, channel_num=1, tag='_'):
    model_t.modify_z = modify_z
    model_t.modify_h = modify_h
    model_t.to(device)
    model_t.eval()
    mse_all_data = torch.empty((0, input_dim_t)).to(device)
    epoch_data_collected = []
    log_likelihood_all_data = []
    all_st = []
    all_guids = []
    with torch.no_grad():
        for j, complete_batched_data_t in tqdm(enumerate(data_loader), total=len(data_loader)):
            batched_data_t = complete_batched_data_t[0]
            guids = complete_batched_data_t[1]
            epochs_nums = complete_batched_data_t[2]
            batched_data_t = batched_data_t.to(device)  # (batch_size, signal_len)
            results_t = model_t(batched_data_t)
            z_latent_t_ = torch.stack(results_t.z_latent, dim=2)  # (batch_size, latent_dim, 150)
            h_hidden_t_ = torch.stack(results_t.hidden_states, dim=2)  # (hidden_layers, batch_size, input_len, h_dim)
            if h_hidden_t_.dim() == 4:
                h_hidden_t__ = h_hidden_t_[-1].permute(0, 2, 1)
            else:
                h_hidden_t__ = h_hidden_t_.permute(0, 2, 1)
            dec_mean_t_ = torch.stack(results_t.decoder_mean, dim=2)  # (batch_size, input_dim, input_size)
            dec_std_t_ = torch.sqrt(torch.exp(torch.stack(results_t.decoder_std, dim=2)))
            Sx_t_ = results_t.Sx.permute(1, 2, 0)  # (batch_size, input_dim, 150)
            enc_mean_t_ = torch.stack(results_t.encoder_mean, dim=2)  # (batch_size, input_dim, 150)
            enc_std_t_ = torch.sqrt(torch.exp(torch.stack(results_t.encoder_std, dim=2)))
            kld_values_t_ = torch.stack(results_t.kld_values, dim=2)

            mse_per_coefficients = torch.sum(((Sx_t_ - dec_mean_t_) ** 2), dim=2) / Sx_t_.size(-1)
            mse_all_data = torch.cat((mse_all_data, mse_per_coefficients), dim=0)
            log_likelihoods = calculate_log_likelihood(dec_mean_t_, dec_std_t_, Sx_t_)
            log_likelihood_all_data.extend(log_likelihoods)
            all_st.append(Sx_t_)
            save_dir = os.path.join(base_dir, tag)
            os.makedirs(save_dir, exist_ok=True)
            signal_channel_dim = Sx_t_.shape[1]
            signal_len = Sx_t_.shape[2]
            for signal_index in range(Sx_t_.shape[0]):
                save_dir_signal = os.path.join(save_dir, f'{guids[signal_index]}')
                os.makedirs(save_dir_signal, exist_ok=True)
                selected_signal = batched_data_t[signal_index]
                sx_selected = Sx_t_[signal_index]  # (input_dim, input_size)
                z_selected = z_latent_t_[signal_index]
                input_data_for_tsne = sx_selected.permute(1, 0).detach().cpu().numpy()
                latent_data_for_tsne = z_selected.permute(1, 0).detach().cpu().numpy()
                tsne = TSNE(n_components=2, random_state=42)
                input_tsne_results = tsne.fit_transform(input_data_for_tsne)
                latent_tsne_results = tsne.fit_transform(latent_data_for_tsne)
                fig, ax = plt.subplots(nrows=2, figsize=(6, 2 * 6 + 3))
                ax[0].scatter(input_tsne_results[:, 0], input_tsne_results[:, 1],
                              c=np.linspace(0, 1, signal_len), cmap='Blues', s=100, edgecolors='black')
                ax[0].set_ylabel('st original')

                ax[1].scatter(latent_tsne_results[:, 0], latent_tsne_results[:, 1],
                              c=np.linspace(0, 1, signal_len), cmap='Reds', s=100, edgecolors='black')
                ax[1].set_ylabel('latent representation')
                plt.savefig(save_dir_signal + '/' + 't-SNE' + '.pdf', bbox_inches='tight',
                            orientation='landscape',
                            dpi=50)
                plt.close(fig)

                if channel_num == 1:
                    signal_c = selected_signal.detach().cpu().numpy()  # for 1 channel
                    two_channel_flag = False
                else:
                    signal_c = selected_signal.squeeze(0).permute(1, 0).detach().cpu().numpy()  # for 2 channel
                    two_channel_flag = True
                plot_averaged_results(signal=signal_c, Sx=sx_selected.detach().cpu().numpy(),
                                      Sxr_mean=dec_mean_t_[signal_index].detach().cpu().numpy(),
                                      Sxr_std=dec_std_t_[signal_index].detach().cpu().numpy(),
                                      z_latent_mean=enc_mean_t_[signal_index].detach().cpu().numpy(),
                                      z_latent_std=enc_std_t_[signal_index].detach().cpu().numpy(),
                                      kld_values=kld_values_t_[signal_index].detach().cpu().numpy(),
                                      h_hidden_mean=h_hidden_t__[signal_index].detach().cpu().numpy(),
                                      plot_latent=True,
                                      plot_klds=True,
                                      two_channel=two_channel_flag,
                                      plot_state=False,
                                      # new_sample=new_sample.detach().cpu().numpy(),
                                      plot_dir=save_dir_signal, tag='')
                plot_scattering_v2(signal=signal_c,
                                   plot_second_channel=two_channel_flag,
                                   Sx=sx_selected.detach().cpu().numpy(), meta=None,
                                   Sxr=dec_mean_t_[signal_index].detach().cpu().numpy(),
                                   Sxr_std=dec_std_t_[signal_index].detach().cpu().numpy(),
                                   z_latent=enc_mean_t_[signal_index].detach().cpu().numpy(),
                                   plot_dir=save_dir_signal, tag='')

    return 0
def run_test(model_t, data_loader, input_dim_t, average_results=False, plot_selected=False,
             modify_h=None, modify_z=None, base_dir=None, channel_num=1, tag='_'):
    model_t.modify_z = modify_z
    model_t.modify_h = modify_h
    model_t.to(device)
    model_t.eval()
    mse_all_data = torch.empty((0, input_dim_t)).to(device)
    epoch_data_collected = []
    log_likelihood_all_data = []
    all_st = []
    all_guids = []
    with torch.no_grad():
        for j, complete_batched_data_t in tqdm(enumerate(data_loader), total=len(data_loader)):
            batched_data_t = complete_batched_data_t[0]
            guids = complete_batched_data_t[1]
            epochs_nums = complete_batched_data_t[2]
            batched_data_t = batched_data_t.to(device)  # (batch_size, signal_len)
            results_t = model_t(batched_data_t)
            z_latent_t_ = torch.stack(results_t.z_latent, dim=2)  # (batch_size, latent_dim, 150)
            h_hidden_t_ = torch.stack(results_t.hidden_states, dim=2)  # (hidden_layers, batch_size, input_len, h_dim)
            if h_hidden_t_.dim() == 4:
                h_hidden_t__ = h_hidden_t_[-1].permute(0, 2, 1)
            else:
                h_hidden_t__ = h_hidden_t_.permute(0, 2, 1)
            dec_mean_t_ = torch.stack(results_t.decoder_mean, dim=2)  # (batch_size, input_dim, input_size)
            dec_std_t_ = torch.sqrt(torch.exp(torch.stack(results_t.decoder_std, dim=2)))
            Sx_t_ = results_t.Sx.permute(1, 2, 0)  # (batch_size, input_dim, 150)
            enc_mean_t_ = torch.stack(results_t.encoder_mean, dim=2)  # (batch_size, input_dim, 150)
            enc_std_t_ = torch.sqrt(torch.exp(torch.stack(results_t.encoder_std, dim=2)))
            kld_values_t_ = torch.stack(results_t.kld_values, dim=2)

            mse_per_coefficients = torch.sum(((Sx_t_ - dec_mean_t_) ** 2), dim=2)/Sx_t_.size(-1)
            mse_all_data = torch.cat((mse_all_data, mse_per_coefficients), dim=0)
            log_likelihoods = calculate_log_likelihood(dec_mean_t_, dec_std_t_, Sx_t_)
            log_likelihood_all_data.extend(log_likelihoods)
            all_st.append(Sx_t_)
            all_guids.extend(guids)
            # if average_results:
            #     tag = tag + 'repeated_and_averaged_'
            #     save_dir = os.path.join(base_dir, tag)
            #     os.makedirs(save_dir, exist_ok=True)
            #     selected_signal = batched_data_t[0]
            #     Sx = Sx_t_[0]
            #     z_latent = torch.mean(z_latent_t_, dim=0)
            #     z_latent_mean = z_latent
            #     z_latent_std = torch.std(z_latent_t_, dim=0)
            #     kld_values = torch.mean(kld_values_t_, dim=0)
            #     dec_mean_mean = torch.mean(dec_mean_t_, dim=0)
            #     dec_mean_std = torch.mean(dec_std_t_, dim=0)
            #     h_hidden = torch.mean(h_hidden_t__, dim=0)
            #     if channel_num == 1:
            #         signal_c = selected_signal.detach().cpu().numpy()  # for 1 channels
            #         two_channel_flag = False
            #     else:
            #         signal_c = selected_signal.squeeze(0).permute(1, 0).detach().cpu().numpy()  # for 2 channels
            #         two_channel_flag = True
            #     plot_averaged_results(signal=signal_c, Sx=Sx.detach().cpu().numpy(),
            #                           Sxr_mean=dec_mean_mean.detach().cpu().numpy(),
            #                           Sxr_std=dec_mean_std.detach().cpu().numpy(),
            #                           z_latent_mean=z_latent_mean.detach().cpu().numpy(),
            #                           z_latent_std=z_latent_std.detach().cpu().numpy(),
            #                           kld_values=kld_values.detach().cpu().numpy(),
            #                           h_hidden_mean=h_hidden.detach().cpu().numpy(),
            #                           plot_latent=True,
            #                           two_channel=two_channel_flag,
            #                           plot_klds=True,
            #                           plot_state=False,
            #                           # new_sample=new_sample.detach().cpu().numpy(),
            #                           plot_dir=save_dir, tag=f'B_')
            #     plot_scattering_v2(signal=signal_c,
            #                        plot_second_channel=two_channel_flag,
            #                        Sx=Sx.detach().cpu().numpy(), meta=None,
            #                        Sxr=dec_mean_mean.detach().cpu().numpy(),
            #                        Sxr_std=dec_mean_std.detach().cpu().numpy(),
            #                        z_latent=z_latent_mean.detach().cpu().numpy(),
            #                        plot_dir=save_dir, tag=f'B_')
        print('stop')
        all_st_tensor = torch.cat(all_st, dim=0)
        all_st_mean = all_st_tensor.mean(dim=0)
        all_st_std = all_st_tensor.std(dim=0)
        tag_hist = tag + 'loglikelihood_'
        save_dir_hist = os.path.join(base_dir, tag_hist)
        os.makedirs(save_dir_hist, exist_ok=True)
        plot_distributions(sx_mean=all_st_mean.detach().cpu().numpy(), sx_std=all_st_std.detach().cpu().numpy(),
                           plot_second_channel=False, plot_sample=False,
                           plot_dir=save_dir_hist, plot_dataset_average=True, tag='st_mean')
        plot_histogram(data=np.array(log_likelihood_all_data), single_channel=True, bins=160, save_dir=save_dir_hist,
                       tag='loglikelihood_original')
        mse_all_data_averaged = torch.mean(mse_all_data, dim=1)
        plot_histogram(data=mse_all_data_averaged.detach().cpu().numpy()/150,
                       single_channel=True,
                       bins=160, save_dir=save_dir_hist, tag='mse-all_dist')
        plot_histogram(data=mse_all_data.detach().cpu().numpy(),
                       single_channel=False,
                       bins=160, save_dir=save_dir_hist, tag='mse-all-data-per')
        return all_st_tensor


if __name__ == '__main__':
    config_file_path = r'config_arguments.yaml'
    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    torch.manual_seed(42)
    np.random.seed(42)
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
    log_file = os.path.join(inference_results_dir, 'log.txt')
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)

    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    print('==' * 50)
    # define model and train it ----------------------------------------------------------------------------------------
    raw_input_size = config['model_config']['VAE_model']['raw_input_size']
    input_size = config['model_config']['VAE_model']['input_size']
    input_dim = config['model_config']['VAE_model']['input_dim']
    latent_dim = config['model_config']['VAE_model']['latent_size']
    n_layers = config['model_config']['VAE_model']['num_RNN_layers']
    rnn_hidden_dim = config['model_config']['VAE_model']['RNN_hidden_dim']
    epochs_num = config['general_config']['epochs']
    lr = config['general_config']['lr']
    channel_num = config['general_config']['channel_num']
    stat_path = os.path.normpath(config['dataset_config']['stat_path'])
    batch_size = config['general_config']['batch_size']['train']

    # hyperparameters
    x_dim = input_dim
    h_dim = rnn_hidden_dim
    z_dim = latent_dim
    n_epochs = epochs_num
    clip = 10
    learning_rate = lr
    batch_size = batch_size  # 128
    # seed = 142
    print_every = 20  # batches
    save_every = 20  # epochs
    plt.ion()
    # Preparing training and testing datasets --------------------------------------------------------------------------
    test_data_dir = config['test_dataset']['test_data_dir']
    test_data_selected_dir = config['test_dataset']['test_data_selected_dir']
    test_hie_data_dir = config['test_dataset']['HIE_test_data_dir']
    checkpoint_dir = config['test_dataset']['test_checkpoint_path']
    if channel_num == 1:
        fhr_healthy_dataset = JsonDatasetPreload(test_data_dir)
        fhr_selected_dataset = JsonDatasetPreload(test_data_selected_dir)
        fhr_hie_dataset = JsonDatasetPreload(test_hie_data_dir)
    else:
        fhr_healthy_dataset = FhrUpPreload(test_data_dir)
        fhr_selected_dataset = FhrUpPreload(test_data_selected_dir)
        fhr_hie_dataset = FhrUpPreload(test_hie_data_dir)

    test_full_dataloader = DataLoader(fhr_healthy_dataset, batch_size=batch_size, shuffle=False)
    test_hie_dataloader = DataLoader(fhr_hie_dataset, batch_size=batch_size, shuffle=False)
    test_selected_dataloader = DataLoader(fhr_selected_dataset, batch_size=batch_size, shuffle=False)

    with open(stat_path, 'rb') as f:
        x_mean = np.load(f)
        x_std = np.load(f)
    log_stat = (x_mean, x_std)

    # create the model and load the checkpoint -------------------------------------------------------------------------
    model = VRNNGauss(input_dim=input_dim, input_size=raw_input_size, h_dim=h_dim, z_dim=z_dim,
                      n_layers=n_layers, device=device, log_stat=log_stat, bias=False)
    params = model.parameters()
    check_point_path = os.path.normpath(checkpoint_dir)
    checkpoint = torch.load(check_point_path)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])

    # testing = plot_tSNE(model_t=model, data_loader=test_selected_dataloader, input_dim_t=input_dim,
    #                     modify_h=None, plot_selected=True, modify_z=None, base_dir=inference_results_dir,
    #                     channel_num=channel_num, tag='t-SNE-')
    #
    # all_st_healthy = run_test(model_t=model, data_loader=test_full_dataloader, input_dim_t=input_dim,
    #                           modify_h=None, plot_selected=True, modify_z=None, base_dir=inference_results_dir,
    #                           channel_num=channel_num, tag='General_test_healthy_')
    #
    # all_st_hie = run_test(model_t=model, data_loader=test_hie_dataloader, input_dim_t=input_dim,
    #                       modify_h=None, plot_selected=True, modify_z=None, base_dir=inference_results_dir,
    #                       channel_num=channel_num, tag='General_test_hie_')
    #
    # for i in range(7):
    #     desired_index = i  # example index
    #     repeated_sample_dataset = RepeatSampleDataset(fhr_selected_dataset, desired_index)
    #     batch_size_repeated = 2000
    #     dataloader = DataLoader(repeated_sample_dataset, batch_size=batch_size_repeated, shuffle=False)
    #     mse_average_repeated = run_test(model_t=model, data_loader=dataloader, input_dim_t=input_dim,
    #                                     average_results=True, modify_h=None, modify_z=None,
    #                                     base_dir=inference_results_dir, channel_num=channel_num,
    #                                     tag=f'repeated_data_{i}_')
    model.to(device)
    g_samples, g_samples_mean, g_samples_sigma = model.generate(input_size=150, batch_size=100)
    g_samples = g_samples.permute(1, 2, 0)
    g_samples_mean = g_samples_mean.permute(1, 2, 0)
    g_samples_sigma = g_samples_sigma.permute(1, 2, 0)
    tag = 'sampled_'
    save_dir_g = os.path.join(inference_results_dir, tag)
    os.makedirs(save_dir_g, exist_ok=True)
    selected_idx = np.random.randint(0, 100, 5)
    for k in selected_idx:
        plot_generated_samples(sx=g_samples[k].detach().cpu().numpy(),
                               sx_mean=g_samples_mean[k].detach().cpu().numpy(),
                               sx_std=g_samples_sigma[k].detach().cpu().numpy(),
                               input_len=input_size, tag=f'test_gen_{k}_', plot_dir=save_dir_g)




    # g_samples, g_samples_mean, g_samples_sigma = model.generate(input_size=150, batch_size=1000)
    # g_samples = g_samples.permute(1, 2, 0)
    # g_samples_mean = g_samples_mean.permute(1, 2, 0)
    # g_samples_sigma = g_samples_sigma.permute(1, 2, 0)
    # tag = 'sampled_'
    # save_dir_g = os.path.join(inference_results_dir, tag)
    # os.makedirs(save_dir_g, exist_ok=True)
    # selected_idx = np.random.randint(0, 100, 5)
    # for k in selected_idx:
    #     plot_distributions(sx_mean=all_st_mean.detach().cpu().numpy(), sx_std=all_st_std.detach().cpu().numpy(),
    #                        plot_dir=save_dir_g, plot_sample=True,
    #                        sample_sx_mean=g_samples_mean[k].detach().cpu().numpy(), tag=f'_{k}_sample')
    #     plot_generated_samples(sx=g_samples[k].detach().cpu().numpy(),
    #                            sx_mean=g_samples_mean[k].detach().cpu().numpy(),
    #                            sx_std=g_samples_sigma[k].detach().cpu().numpy(),
    #                            input_len=input_size, tag=f'test_gen_{k}_', plot_dir=save_dir_g)

    # # for m in
    # all_sample_loglikelihood = []
    # for m in range(g_samples.shap[0]):
    #     calculate_log_likelihood_generated(g_samples, g_samples_mean, g_samples_sigma)
    #
    # fig, ax = plt.subplots(figsize=(32, 6))
    # ax.hist(data, bins=160, color='royalblue', alpha=0.99, edgecolor='black')
    # # ax.plot(data, np.full_like(data, -0.01), '|', color='black', alpha=0.1, markersize=10)
    # ax.set_xlabel('Value', fontsize=14)
    # ax.set_ylabel('Frequency', fontsize=14)
    # ax.set_title('Distribution of Data with Histogram', fontsize=16)
    # ax.grid(True, linewidth=0.1)  # Reduce the grid line thickness
    # mean = np.mean(data)
    # median = np.median(data)
    # # ax.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
    # # ax.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.2f}')
    # ax.legend()
    # plt.savefig(save_dir_g + '/' + 'loglikelihood' + '_distribution' + '.pdf', bbox_inches='tight',
    #             orientation='landscape', dpi=50)
    # plt.close(fig)
