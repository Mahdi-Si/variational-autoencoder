import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import torch
import torch.nn as nn
import torchvision
import numpy as np
from matplotlib.colors import LogNorm

from kymatio.torch import Scattering1D
import pickle
from scipy.signal import decimate
import plotly.graph_objects as go
import os

def calculate_stats(loader):
    mean = 0.0
    std = 0.0
    total_samples = 0

    for data in loader:
        # Assuming data shape is [batch_size, channels, ...]
        data = data.unsqueeze(1)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean, std


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


def plot_scattering(signal=None, plot_order=None, Sx=None, meta=None,
                    Sxr=None, plot_dir=None, tag=''):
    if torch.is_tensor(signal) and signal.is_cuda:
        signal = signal.cpu().detach().numpy()
    if torch.is_tensor(Sx) and Sx.is_cuda:
        Sx = Sx.cpu().detach().numpy()
    Fs = 4
    Q = 1
    J = 11
    Over = 0
    T = 2 ** (J - 7)
    N_CHAN = 12
    log_eps = 1e-3
    dtype = np.float32
    SINGLE_BATCH_SIZE = 1
    N = len(signal)
    if Sxr is not None:
        # N_ROWS = 3
        N_ROWS = len(plot_order) + 4
    else:
        # N_ROWS = 2
        N_ROWS = len(plot_order) + 1

    t_in = np.arange(0, N) / Fs

    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 8, 'axes.labelsize': 8})
    i_row = 0

    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(14, 16),
                           gridspec_kw={"width_ratios": [40, 1]})
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, signal, linewidth=0.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')


    for order in plot_order:
        if isinstance(order, int):
            i_row += 1
            order_i = np.where(meta['order'] == order)
            x = Sx[:, order_i, :].squeeze()
            if order == 0:
                ax[i_row, 0].plot(x.squeeze(), linewidth=0.5)
                ax[i_row, 1].set_axis_off()
            else:
                imgplot = ax[i_row, 0].imshow(np.log(x + log_eps), aspect='auto',
                                              extent=[0, N / Fs, Sx.shape[0], 0])
                # imgplot = ax[i_row, 0].imshow(x, aspect='auto')
                ax[i_row, 1].set_axis_on()
                fig.colorbar(imgplot, cax=ax[i_row, 1])
            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            ax[i_row, 0].set_xticklabels([])
            ax[i_row, 0].set_ylabel(f'Order {order}')
        elif isinstance(order, tuple):
            i_row += 1
            order_i = np.where(np.isin(meta['order'], order))
            x = Sx[:, order_i, :].squeeze()
            imgplot = ax[i_row, 0].imshow(np.log(x + log_eps), aspect='auto',
                                          extent=[0, N / Fs, Sx.shape[0], 0])
            # imgplot = ax[i_row, 0].imshow(x, aspect='auto')
            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            ax[i_row, 0].set_xticklabels([])
            ax[i_row, 0].set_ylabel(f'Order {order}')
            ax[i_row, 1].set_axis_on()
            fig.colorbar(imgplot, cax=ax[i_row, 1])

    if Sxr is not None:
        i_row += 1
        if torch.is_tensor(Sx) and Sxr.is_cuda:
            Sxr = Sxr.cpu().detach().numpy()
        Sxr = Sxr.transpose(1, 0)

        ax[i_row, 0].plot(Sxr[0, :], linewidth=0.5)
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].autoscale(enable=True, axis='x reconstructed', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('Reconstructed order 0')

        i_row += 1
        imgplot = ax[i_row, 0].imshow(np.log(Sxr[1:, :] + log_eps), aspect='auto',
                                      extent=[0, N / Fs, Sxr.shape[0], 0])
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('Reconstructed order 1')
        ax[i_row, 1].set_axis_on()
        fig.colorbar(imgplot, cax=ax[i_row, 1])

        i_row += 1
        imgplot = ax[i_row, 0].imshow(np.log(Sxr + log_eps), aspect='auto',
                                      extent=[0, N / Fs, Sxr.shape[0], 0])
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('Reconstructed order 1 and 0')
        ax[i_row, 1].set_axis_on()
        fig.colorbar(imgplot, cax=ax[i_row, 1])

    cmstr = 'bwr'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    plt.savefig(plot_dir + '/' + tag + '_' + '.png', bbox_inches='tight', orientation='landscape', dpi=900)
    plt.close(fig)


def plot_original_reconstructed(original_x, reconstructed_x, plot_dir=None, tag=''):
    import matplotlib as mpl
    # Set the font globally to Times New Roman, size 18
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 14

    fig, ax = plt.subplots(3, 1, figsize=(30, 7))
    ax[0].plot(original_x, label='Original')
    ax[1].plot(reconstructed_x, label='Reconstructed')
    ax[2].plot(original_x, label='Original', linewidth=2, color='#474747')
    ax[2].plot(reconstructed_x, label='Reconstructed', linewidth=1.5, color='#43AA8B')

    # Adding legends
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    # Setting x and y labels
    ax[0].set_xlabel('Sample')
    ax[0].set_ylabel('FHR')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Reconstructed FHR')

    # Showing grid
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.savefig(plot_dir + '/' + tag + '_' + '_st.png', bbox_inches='tight', orientation='landscape')
    plt.close(fig)


def plot_scattering_v2(signal=None, plot_order=None, Sx=None, meta=None,
                       Sxr=None, Sxr_std=None, z_latent=None, plot_dir=None, tag=''):
    Fs = 4
    log_eps = 1e-3
    N = len(signal)
    # if Sxr is not None:
    #     # N_ROWS = 3
    #     N_ROWS = len(plot_order) + 4
    # else:
    #     # N_ROWS = 2
    #     N_ROWS = len(plot_order) + 1
    N_ROWS = 4 + (Sx.shape[0])
    t_in = np.arange(0, N) / Fs
    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 8, 'axes.labelsize': 8})
    i_row = 0

    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(20, 56),
                           gridspec_kw={"width_ratios": [60, 1]})
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, signal, linewidth=0.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(Sx, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, Sx.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('True ST')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(Sxr, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, Sx.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Reconstructed ST')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(z_latent, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, z_latent.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Latent Representation')

    for i in range(Sx.shape[0]):
        i_row += 1
        ax[i_row, 0].plot(Sx[i, :], linewidth=0.9, label="True")
        ax[i_row, 0].plot(Sxr[i, :], linewidth=0.5, label="Reconstructed")
        if Sxr_std is not None:
            ax[i_row, 0].fill_between(np.arange(len(Sxr_std[i, :])),
                                      Sxr[i, :] - Sxr_std[i, :],
                                      Sxr[i, :] + Sxr_std[i, :],
                                      color='blue', alpha=0.2, label='Std dev')
        ax[i_row, 0].legend()
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].set_ylabel(f'Coefficient {i}')


    cmstr = 'bwr'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    plt.savefig(plot_dir + '/' + tag + '_' + '.png', bbox_inches='tight', orientation='landscape', dpi=100)
    plt.close(fig)


def plot_loss_dict(loss_dict, epoch_num, plot_dir):
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.size"] = 32
    # plt.rcParams['text.usetex'] = True
    num_rows = len(loss_dict.keys())
    t = np.arange(1, epoch_num + 1)
    # fig, ax = plt.subplots(nrows=num_rows, ncols=1, figsize=(15, 30))
    fig = go.Figure()
    for i, (key, val) in enumerate(loss_dict.items()):
        fig.add_trace(go.Scatter(y=val, mode='lines', name=key))
        # ax[i].autoscale(enable=True, axis='x', tight=True)
        # ax[i].plot(t, val, label=key, color='#265073', linewidth=0.7)
        # ax[i].set_ylabel(key, fontsize=14)
        # ax[i].grid()
    # fig = go.Figure()
    # Update layout to add titles and adjust other settings as needed
    fig.update_layout(title='Loss',
                      xaxis_title='Epoch',
                      yaxis_title='Loss',
                      legend_title='Legend',
                      template='plotly_dark')

    # Save the figure as an HTML file
    fig_path = os.path.join(plot_dir, 'loss_plot.html')
    fig.write_html(fig_path)
    # plt.savefig(f'{plot_dir}/Loss_st.png', bbox_inches='tight', dpi=100)


def plot_averaged_results(signal=None, Sx=None, Sxr_mean=None, Sxr_std=None, z_latent_mean=None,
                          z_latent_std=None, kld_values=None, plot_dir=None, tag=''):
    Fs = 4
    log_eps = 1e-3
    N = len(signal)
    N_ROWS = 5 + (z_latent_mean.shape[0])
    t_in = np.arange(0, N) / Fs
    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 8, 'axes.labelsize': 8})
    i_row = 0

    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(20, 56),
                           gridspec_kw={"width_ratios": [60, 1]})
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, signal, linewidth=0.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')

    i_row += 1
    z_diff = np.diff(z_latent_mean, axis=1)
    z_diff_squared_sum = np.square(z_diff).sum(axis=0)
    # z_diff_sum = z_diff.sum(axis=0)
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(z_diff_squared_sum, linewidth=0.8)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Latent Z difference')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(kld_values, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, kld_values.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('KLD')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(z_latent_mean, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, z_latent_mean.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Latent Representation Mean')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(z_latent_std, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, z_latent_mean.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Latent Representation Std')


    for i in range(z_latent_mean.shape[0]):
        i_row += 1
        ax[i_row, 0].plot(z_latent_mean[i, :], linewidth=1.9, label="Latent Representation")
        ax[i_row, 0].fill_between(np.arange(len(z_latent_mean[i, :])),
                                  z_latent_mean[i, :] - z_latent_std[i, :],
                                  z_latent_mean[i, :] + z_latent_std[i, :],
                                  color='blue', alpha=0.25, label='Std dev')
        ax[i_row, 0].legend()
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].set_ylabel(f'Coefficient {i}')


    cmstr = 'bwr'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    plt.savefig(plot_dir + '/' + tag + '_' + '.png', bbox_inches='tight', orientation='landscape', dpi=100)
    plt.close(fig)


