
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
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np
from Variational_AutoEncoder.datasets.custom_datasets import JsonDatasetPreload, RepeatSampleDataset, FhrUpPreload
from Variational_AutoEncoder.utils.data_utils import plot_scattering_v2, plot_averaged_results, plot_general_mse, \
    plot_generated_samples

# from vrnn_gauss import VRNN_Gauss
from vrnn_gauss_experiment_5_old import VRNNGauss
from Variational_AutoEncoder.utils.run_utils import log_resource_usage, StreamToLogger, setup_logging
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # remove this line when creating a new environment

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')


def calculate_log_likelihood(sample, sample_mu, sample_sigma):
    # Assuming sample_mu and sample_sigma are the means and standard deviations of the distributions
    # from which the samples were drawn. Note: sample_sigma should be standard deviations, not variances.

    # Create a Normal distribution object with the mean and std from the generator
    dist = torch.distributions.Normal(sample_mu, sample_sigma)

    # Compute log probability of the generated sample under the generated distribution parameters
    log_probs = dist.log_prob(sample)

    # Sum the log probabilities over all dimensions and samples
    total_log_likelihood = log_probs.sum()

    return total_log_likelihood


def run_test(model_t, data_loader, input_dim_t, average_results=False, plot_selected=False,
             modify_h=None, modify_z=None, base_dir=None, channel_num=1, tag='_'):
    model_t.modify_z = modify_z
    model_t.modify_h = modify_h
    model_t.to(device)
    model_t.eval()
    mse_all_data = torch.empty((0, input_dim_t)).to(device)
    collected_data = []
    epoch_data_collected = []
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
            # h_didden_t__ (batch_size, input_dim, input_size)
            # h_hidden_t__ = torch.sum(h_hidden_t_, dim=0).permute(0, 2, 1)
            dec_mean_t_ = torch.stack(results_t.decoder_mean, dim=2)  # (batch_size, input_dim, input_size)
            dec_std_t_ = torch.sqrt(torch.exp(torch.stack(results_t.decoder_std, dim=2)))
            Sx_t_ = results_t.Sx.permute(1, 2, 0)  # (batch_size, input_dim, 150)
            enc_mean_t_ = torch.stack(results_t.encoder_mean, dim=2)  # (batch_size, input_dim, 150)
            enc_std_t_ = torch.sqrt(torch.exp(torch.stack(results_t.encoder_std, dim=2)))
            kld_values_t_ = torch.stack(results_t.kld_values, dim=2)

            mse_coefficients = torch.sum(((Sx_t_ - dec_mean_t_) ** 2), dim=2)/Sx_t_.size(-1)
            mse_all_data = torch.cat((mse_all_data, mse_coefficients), dim=0)
            max_values = torch.max(h_hidden_t__, dim=2)
            min_values = torch.min(h_hidden_t__, dim=2)
            average_values = torch.mean(h_hidden_t__, dim=2)
            std_values = torch.std(h_hidden_t__, dim=2)
            diff_values = torch.diff(h_hidden_t__, dim=2)

            batch_results = zip(guids, epochs_nums, mse_coefficients.cpu().numpy(),
                                max_values.values.cpu().numpy(),
                                min_values.values.cpu().numpy(),
                                average_values.detach().cpu().numpy(),
                                std_values.detach().cpu().numpy(),
                                diff_values.detach().cpu().numpy())
            collected_data.extend([(guid, epoch.item(), mse_val, max_val, min_val, average_val, std_val, diff_val)
                                   for guid, epoch, mse_val, max_val, min_val, average_val, std_val, diff_val
                                   in batch_results])

            for m in range(epochs_nums.shape[0]):
                data_dict = {'guid': guids[m], 'epoch': epochs_nums[m].item()}
                for n in range(mse_coefficients.shape[1]):
                    data_dict[f'mse_coefficients_{n}'] = mse_coefficients.cpu().numpy()[m, n]
                    data_dict[f'max_{n}'] = max_values.values.cpu().numpy()[m, n]
                    data_dict[f'min_{n}'] = min_values.values.cpu().numpy()[m, n]
                    data_dict[f'mean_{n}'] = average_values.detach().cpu().numpy()[m, n]
                    data_dict[f'std_{n}'] = std_values.detach().cpu().numpy()[m, n]

                epoch_data_collected.append(data_dict)
            if average_results:
                tag = tag + 'repeated_and_averaged_'
                save_dir = os.path.join(base_dir, tag)
                os.makedirs(save_dir, exist_ok=True)
                selected_signal = batched_data_t[0]
                Sx = Sx_t_[0]
                z_latent = torch.mean(z_latent_t_, dim=0)
                z_latent_mean = z_latent
                z_latent_std = torch.std(z_latent_t_, dim=0)
                kld_values = torch.mean(kld_values_t_, dim=0)
                dec_mean_mean = torch.mean(dec_mean_t_, dim=0)
                dec_mean_std = torch.mean(dec_std_t_, dim=0)
                h_hidden = torch.mean(h_hidden_t__, dim=0)
                if channel_num == 1:
                    signal_c = selected_signal.detach().cpu().numpy()  # for 1 channels
                    two_channel_flag = False
                else:
                    signal_c = selected_signal.squeeze(0).permute(1, 0).detach().cpu().numpy()  # for 2 channels
                    two_channel_flag = True
                plot_averaged_results(signal=signal_c, Sx=Sx.detach().cpu().numpy(),
                                      Sxr_mean=dec_mean_mean.detach().cpu().numpy(),
                                      Sxr_std=dec_mean_std.detach().cpu().numpy(),
                                      z_latent_mean=z_latent_mean.detach().cpu().numpy(),
                                      z_latent_std=z_latent_std.detach().cpu().numpy(),
                                      kld_values=kld_values.detach().cpu().numpy(),
                                      h_hidden_mean=h_hidden.detach().cpu().numpy(),
                                      plot_latent=True,
                                      two_channel=two_channel_flag,
                                      plot_klds=True,
                                      plot_state=False,
                                      # new_sample=new_sample.detach().cpu().numpy(),
                                      plot_dir=save_dir, tag=f'B_')
                plot_scattering_v2(signal=signal_c,
                                   plot_second_channel=two_channel_flag,
                                   Sx=Sx.detach().cpu().numpy(), meta=None,
                                   Sxr=dec_mean_mean.detach().cpu().numpy(),
                                   Sxr_std=dec_mean_std.detach().cpu().numpy(),
                                   z_latent=z_latent_mean.detach().cpu().numpy(),
                                   plot_dir=save_dir, tag=f'B_')

            else:
                if plot_selected:
                    tag = tag + 'selected_sequences_'
                    save_dir = os.path.join(base_dir, tag)
                    os.makedirs(save_dir, exist_ok=True)
                    selected_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                    # selected_idx = np.arange(len(batched_data_t))
                    for idx in selected_idx:
                        selected_signal = batched_data_t[idx]
                        guid_selected = guids[idx]
                        Sx = Sx_t_[idx]  # might need to permute (1, 0)
                        z_latent = z_latent_t_[idx]
                        # z_latent = h_hidden__[idx]
                        z_latent_mean = z_latent
                        z_latent_std = enc_std_t_[idx]
                        kld_values = kld_values_t_[idx]
                        dec_mean_mean = dec_mean_t_[idx]
                        dec_mean_std = dec_std_t_[idx]
                        h_hidden = h_hidden_t__[idx]
                        if channel_num == 1:
                            signal_c = selected_signal.detach().cpu().numpy()  # for 1 channel
                            two_channel_flag = False
                        else:
                            signal_c = selected_signal.squeeze(0).permute(1, 0).detach().cpu().numpy()  # for 2 channel
                            two_channel_flag = True
                        plot_averaged_results(signal=signal_c, Sx=Sx.detach().cpu().numpy(),
                                              Sxr_mean=dec_mean_mean.detach().cpu().numpy(),
                                              Sxr_std=dec_mean_std.detach().cpu().numpy(),
                                              z_latent_mean=z_latent_mean.detach().cpu().numpy(),
                                              z_latent_std=z_latent_std.detach().cpu().numpy(),
                                              kld_values=kld_values.detach().cpu().numpy(),
                                              h_hidden_mean=h_hidden.detach().cpu().numpy(),
                                              plot_latent=True,
                                              plot_klds=True,
                                              two_channel=two_channel_flag,
                                              plot_state=False,
                                              # new_sample=new_sample.detach().cpu().numpy(),
                                              plot_dir=save_dir, tag=f'{guid_selected}_{j}_{idx}_')
                        plot_scattering_v2(signal=signal_c,
                                           plot_second_channel=two_channel_flag,
                                           Sx=Sx.detach().cpu().numpy(), meta=None,
                                           Sxr=dec_mean_mean.detach().cpu().numpy(),
                                           Sxr_std=dec_mean_std.detach().cpu().numpy(),
                                           z_latent=z_latent_mean.detach().cpu().numpy(),
                                           plot_dir=save_dir, tag=f'{guid_selected}_{j}_{idx}_')
        # mse_all_data (dataset_size, input_dim)
        # plot_general_mse(all_mse=mse_all_data.permute(1, 0).detach().cpu().numpy(),
        #                  tag=f'mses_{tag}',
        #                  plot_dir=base_dir)
        mse_average = mse_all_data.mean(dim=0)
        print('==' * 50)
        print(f'MSE each dim: {mse_average}')
        print(f'MSE average: {mse_average.sum()}')
        print('==' * 50)
        df_pre = pd.DataFrame(collected_data, columns=['guid', 'epoch', 'MSE', 'Max', 'Min', 'Average', 'Std', 'Diff'])
        df_pre.to_csv(f'{base_dir}/{tag}.csv')
        df_all_data = pd.DataFrame(epoch_data_collected)
        df_all_data.to_csv(f'{base_dir}/{tag}_all_data.csv')
        return mse_average


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
    # dataset_dir = os.path.normpath(config['dataset_config']['dataset_dir'])
    # dataset_dir = os.path.normpath(config['dataset_config']['aux_dataset_dir'])
    # aux_dataset_hie_dir = os.path.normpath(config['dataset_config']['aux_dataset_dir'])
    dataset_dir = r"C:\Users\mahdi\Desktop\Mahdi-Si-Projects\AI\datasets\FHR\Json\selected_one_jason"
    if channel_num == 1:
        fhr_healthy_dataset = JsonDatasetPreload(dataset_dir)
    else:
        fhr_healthy_dataset = FhrUpPreload(dataset_dir)  # two channel dataset
    # fhr_aux_hie_dataset = JsonDatasetPreload(aux_dataset_hie_dir)
    data_loader_healthy = DataLoader(fhr_healthy_dataset, batch_size=batch_size, shuffle=False)
    # data_loader_hie = DataLoader(fhr_aux_hie_dataset, batch_size=batch_size, shuffle=False)

    with open(stat_path, 'rb') as f:
        x_mean = np.load(f)
        x_std = np.load(f)
    log_stat = (x_mean, x_std)
    # fhr_healthy_dataset = FHRDataset(healthy_list)

    # model = VRNN(x_len=raw_input_size, x_dim=x_dim, h_dim=h_dim, z_dim=z_dim, n_layers=n_layers, log_stat=log_stat)

    # create the model and load the checkpoint -------------------------------------------------------------------------
    model = VRNNGauss(input_dim=input_dim, input_size=raw_input_size, h_dim=h_dim, z_dim=z_dim,
                      n_layers=n_layers, device=device, log_stat=log_stat, bias=False)
    params = model.parameters()
    check_point_path = os.path.normpath(
        r"C:\Users\mahdi\Desktop\Mahdi-Si-Projects\AI\runs\variational-autoencoder\VM\h96_l11\VRNN-5697.pth"
    )
    checkpoint = torch.load(check_point_path)
    # model.load_state_dict(checkpoint)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])
    mse_average = run_test(model_t=model, data_loader=data_loader_healthy, input_dim_t=input_dim, modify_h=None,
                           plot_selected=True, modify_z=None, base_dir=inference_results_dir, channel_num=channel_num,
                           tag='General_test_')
    columns = [f'St_coefficient_{i}' for i in range(0, input_dim)]
    columns.insert(0, 'Changed')
    df = pd.DataFrame(columns=columns)
    # df_2 = pd.DataFrame(columns=columns)
    # final_list = ["No Change"] + mse_average.cpu().tolist()
    # df.loc[len(df)] = final_list
    # df_2.loc[len(df_2)] = final_list
    # for i in range(latent_dim):
    #     modify_dims_z = list(range(0, latent_dim))
    #     scale = [int(x) for x in np.ones(latent_dim)]
    #     # scale = np.ones(latent_dim).astype(int).tolist()
    #     scale[i] = int(10)
    #     shift = [int(x) for x in np.zeros(latent_dim)]
    #     modify_z_dict = {'modify_dims': modify_dims_z, 'scale': scale, 'shift': shift}
    #     print(f'modified z {i}: \n {modify_z_dict}')
    #     print('=='*50)
    #
    #     mse_er = run_test(model_t=model, data_loader=data_loader_healthy, input_dim_t=input_dim, modify_h=None,
    #                       modify_z=modify_z_dict, plot_selected=True, base_dir=inference_results_dir,
    #                       channel_num=channel_num, tag=f'latent_dim_{i}_')
    #
    #     final_list = [f'Latent_dim_{i}'] + mse_er.cpu().tolist()
    #     df.loc[len(df)] = final_list
    #
    # df.to_csv((inference_results_dir + '/' + 'mse_losses_latent_dims.csv'), index=False)

    #
    # for i in range(0, rnn_hidden_dim):
    #     modify_dims_h = [int(i)]
    #     scale = [int(0)]
    #     shift = [int(0)]
    #     modify_h_dict = {'modify_dims': modify_dims_h, 'scale': scale, 'shift': shift}
    #     print(f'modified z {i}: \n {modify_h_dict}')
    #     print('=='*50)
    #
    #     mse_average = run_test(model_t=model, data_loader=data_loader_healthy, input_dim_t=input_dim, modify_z=None,
    #                            modify_h=modify_h_dict,
    #                            base_dir=inference_results_dir, tag=f'hidden_{i}')
    #     final_list = [f'Hidden_dim_{i}'] + mse_average.cpu().tolist()
    #     df_2.loc[len(df_2)] = final_list
    # df_2.to_csv((inference_results_dir + '/' + 'mse_losses_hidden_dims.csv'), index=False)
    # repeated sample simulations --------------------------------------------------------------------------------------
    for i in range(7):
        desired_index = i  # example index
        repeated_sample_dataset = RepeatSampleDataset(fhr_healthy_dataset, desired_index)

        batch_size_repeated = 2000
        dataloader = DataLoader(repeated_sample_dataset, batch_size=batch_size_repeated, shuffle=False)
        mse_average_repeated = run_test(model_t=model, data_loader=dataloader, input_dim_t=input_dim,
                                        average_results=True, modify_h=None, modify_z=None,
                                        base_dir=inference_results_dir, channel_num=channel_num,
                                        tag=f'repeated_data_{i}_')

    g_samples, g_samples_mean, g_samples_sigma = model.generate(input_size=150, batch_size=1000)
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
