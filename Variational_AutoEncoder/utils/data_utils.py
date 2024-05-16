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
    ax[i_row, 0].plot(t_in, signal, linewidth=1.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')


    for order in plot_order:
        if isinstance(order, int):
            i_row += 1
            order_i = np.where(meta['order'] == order)
            x = Sx[:, order_i, :].squeeze()
            if order == 0:
                ax[i_row, 0].plot(x.squeeze(), linewidth=1.5)
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

        ax[i_row, 0].plot(Sxr[0, :], linewidth=1.5)
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
    plt.savefig(plot_dir + '/' + tag + '_' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
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
    plt.savefig(plot_dir + '/' + tag + '_' + '_st.pdf', bbox_inches='tight', orientation='landscape')
    plt.close(fig)


def plot_scattering_v2(signal=None, plot_order=None, Sx=None, meta=None, plot_second_channel=False,
                       Sxr=None, Sxr_std=None, z_latent=None, plot_dir=None, tag=''):
    """
    Plots the results of the model.
    :param signal: (signal_size, 1)
    :param plot_order:
    :param Sx: (input_dim, input_size)
    :param meta:
    :param Sxr: (input_dim, input_size)
    :param Sxr_std: (input_dim, input_size)
    :param z_latent: (latent_dim, latent_size)
    :param plot_dir:
    :param tag:
    :return:
    """
    Fs = 4
    log_eps = 1e-3
    N = len(signal)
    # if Sxr is not None:
    #     # N_ROWS = 3
    #     N_ROWS = len(plot_order) + 4
    # else:
    #     # N_ROWS = 2
    #     N_ROWS = len(plot_order) + 1
    if plot_second_channel:
        N_ROWS = 5 + (Sx.shape[0])
        signal_1 = signal[:, 0]
        signal_2 = signal[:, 1]
    else:
        N_ROWS = 4 + (Sx.shape[0])
        signal_2 = signal
        signal_1 = signal
    t_in = np.arange(0, N) / Fs
    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})

    i_row = 0

    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                           gridspec_kw={"width_ratios": [80, 1]})
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, signal_1, linewidth=1.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')

    if plot_second_channel:
        i_row += 1
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(t_in, signal_2, linewidth=1.5)
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('UP')

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
        ax[i_row, 0].plot(Sx[i, :], linewidth=1.5, label="True")
        ax[i_row, 0].plot(Sxr[i, :], linewidth=1, label="Reconstructed")
        if Sxr_std is not None:
            ax[i_row, 0].fill_between(np.arange(len(Sxr_std[i, :])),
                                      Sxr[i, :] - Sxr_std[i, :],
                                      Sxr[i, :] + Sxr_std[i, :],
                                      color='blue', alpha=0.1, label='Std dev')
        ax[i_row, 0].legend()
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].set_ylabel(f'Coefficient {i}')


    cmstr = 'bwr'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    plt.savefig(plot_dir + '/' + tag + '_' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
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
    # plt.savefig(f'{plot_dir}/Loss_st.pdf', bbox_inches='tight', dpi=50)


def plot_averaged_results(signal=None, Sx=None, Sxr_mean=None, Sxr_std=None, z_latent_mean=None, h_hidden_mean=None,
                          h_hidden_std=None, z_latent_std=None, kld_values=None, plot_dir=None, new_sample=None,
                          plot_latent=False, plot_klds=False, plot_state=False, two_channel=False, tag=''):
    Fs = 4
    log_eps = 1e-3

    if two_channel:
        signal_1 = signal[:, 0]
        signal_2 = signal[:, 1]
    else:
        signal_2 = signal
        signal_1 = signal

    N = len(signal_1)

    N_ROWS = 7 + (z_latent_mean.shape[0])
    t_in = np.arange(0, N) / Fs
    cmstr = 'Blues'
    # cmstr = 'viridis'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})
    i_row = 0
    # plot st vs reconstructed st, kld, hidden sates and each latent variable ------------------------------------------
    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                           gridspec_kw={"width_ratios": [60, 1]})

    # plot true fhr
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, signal_1, linewidth=1.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')
    if two_channel:
        ax2 = ax[i_row, 0].twinx()
        ax2.plot(t_in, signal_2, linewidth=1.5, color="#c96b00")
        ax2.set_ylabel('UP')

    # plot latent z difference
    i_row += 1
    z_diff = np.diff(z_latent_mean, axis=1)
    z_diff_squared_sum = np.square(z_diff).sum(axis=0)
    # z_diff_sum = z_diff.sum(axis=0)
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(z_diff_squared_sum, linewidth=1.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Latent Z difference')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(Sx, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, Sx.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('True ST')

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

    if h_hidden_mean is not None:
        i_row += 1
        imgplot = ax[i_row, 0].imshow(h_hidden_mean, aspect='auto', norm="symlog",
                                      extent=[0, N / Fs, h_hidden_mean.shape[0], 0])
        ax[i_row, 1].set_axis_on()
        fig.colorbar(imgplot, cax=ax[i_row, 1])
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('Hidden States Mean')

    for i in range(z_latent_mean.shape[0]):
        i_row += 1
        ax[i_row, 0].plot(z_latent_mean[i, :], linewidth=1.5, label="Latent Representation")
        ax[i_row, 0].fill_between(np.arange(len(z_latent_mean[i, :])),
                                  z_latent_mean[i, :] - z_latent_std[i, :],
                                  z_latent_mean[i, :] + z_latent_std[i, :],
                                  color='blue', alpha=0.25, label='Std dev')
        ax[i_row, 0].legend()
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].set_ylabel(f'Coefficient {i}')

    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    plt.savefig(plot_dir + '/' + tag + 'overall' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=300)
    plt.close(fig)
    # ------------------------------------------------------------------------------------------------------------------
    # plot latent dim and histogram of it
    if plot_latent:
        i_row = 0
        N_ROWS = z_latent_mean.shape[0] + 2
        # N_ROWS = 1 * z_latent_mean.shape[0] + z_latent_mean.shape[0] + 2
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                               gridspec_kw={"width_ratios": [60, 1]})
        t_original = np.linspace(0, 1, len(signal_1))
        t_reduced = np.linspace(0, 1, Sx.shape[1])

        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(t_original, signal_1, linewidth=1.5, color="#3D8361")
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel(f'FHR')
        if two_channel:
            ax2 = ax[i_row, 0].twinx()
            ax2.plot(t_original, signal_2, linewidth=1.5, color="#135D66")
            ax2.set_ylabel('UP')
        # for j in range(Sx.shape[0]):
        for i in range(z_latent_mean.shape[0]):
            i_row += 1
            ax[i_row, 1].set_axis_off()
            ax2 = ax[i_row, 0].twinx()
            ax2.plot(t_original, signal_1, linewidth=1, color="#0C2D57")
            marker_line, stem_lines, baseline = ax[i_row, 0].stem(t_reduced, 1*z_latent_mean[i, :], basefmt=" ")
            plt.setp(stem_lines, 'color', "#FC6736", 'linewidth', 2)
            plt.setp(marker_line, 'color', "#387ADF")
            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            ax[i_row, 0].set_xticklabels([])
            ax[i_row, 0].set_ylabel(f'Latent Dim {i} Coefficient')

        i_row += 1
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(Sx[0, :], linewidth=1.5, color="#0C2D57")
        marker_line, stem_lines, baseline = ax[i_row, 0].stem(1*np.mean(z_latent_mean, axis=0), basefmt=" ")
        plt.setp(stem_lines, 'color', "#FC6736", 'linewidth', 2)
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel(f'Latent Dim Averaged')

        # for i in range(z_latent_mean.shape[0]):
        #     i_row += 1
        #     ax[i_row, 1].set_axis_off()
        #     ax[i_row, 0].hist(z_latent_mean[i, :], bins=40, alpha=0.6, color='blue', rwidth=0.9, edgecolor='black')
        #     ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        #     # ax[i_row, 0].set_xticklabels([])
        #     ax[i_row, 0].set_ylabel(f'Latent Dim Histogram {i}')

        plt.savefig(plot_dir + '/' + tag + '_latent' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
        plt.close(fig)
    # ------------------------------------------------------------------------------------------------------------------

    # plot hidden dims -------------------------------------------------------------------------------------------------
    if plot_state:
        i_row = 0
        # N_ROWS = 1 * h_hidden_mean.shape[0] + h_hidden_mean.shape[0] * Sx.shape[0] + 2
        N_ROWS = 1 * h_hidden_mean.shape[0] + h_hidden_mean.shape[0] * 1 + 2
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                               gridspec_kw={"width_ratios": [60, 1]})
        t_original = np.linspace(0, 1, len(signal_1))
        t_reduced = np.linspace(0, 1, Sx.shape[1])

        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(t_original, signal_1, linewidth=1.5, color="#3D8361")
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel(f'FHR')
        if two_channel:
            ax2 = ax[i_row, 0].twinx()
            ax2.plot(t_in, signal_2, linewidth=1.5, color="#135D66")
            ax2.set_ylabel('UP')
        # for j in range(Sx.shape[0]):
        for i in range(h_hidden_mean.shape[0]):
            i_row += 1
            ax[i_row, 1].set_axis_off()
            # ax2 = ax[i_row, 0].twinx()
            # ax2.plot(t_reduced, Sx[0, :], linewidth=1, color="#0C2D57")

            ax3 = ax[i_row, 0].twinx()
            ax3.plot(t_original, signal_1, linewidth=1.5, color="#3D8361")

            # marker_line, stem_lines, baseline = ax[i_row, 0].stem(t_reduced, 1*h_hidden_mean[i, :], basefmt=" ")
            # plt.setp(stem_lines, 'linewidth', 0)
            # plt.setp(marker_line, 'color', "#FC6736", 'marker', 'o', 'markersize', 8)
            ax[i_row, 0].plot(t_reduced, 1*h_hidden_mean[i, :], linewidth=1.5, color="#FC6736", marker='o',
                              linestyle='-', markersize=8)

            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            ax[i_row, 0].set_xticklabels([])
            ax[i_row, 0].set_ylabel(f'Latent Dim {i} Coefficient 0')

        i_row += 1
        ax[i_row, 1].set_axis_off()

        marker_line, stem_lines, baseline = ax[i_row, 0].stem(t_reduced, 1*np.mean(h_hidden_mean, axis=0), basefmt=" ")
        plt.setp(stem_lines, 'color', "#FC6736", 'linewidth', 0)
        plt.setp(marker_line, 'color', "#FC6736", 'marker', 'o', 'markersize', 8)
        ax2 = ax[i_row, 0].twinx()
        ax2.plot(t_reduced, Sx[0, :], linewidth=1.5, color="#0C2D57")
        ax3 = ax[i_row, 0].twinx()
        ax3.plot(t_original, signal_1, linewidth=1.5, color="#3D8361")
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel(f'Hidden Dim Averaged')

        for i in range(h_hidden_mean.shape[0]):
            i_row += 1
            ax[i_row, 1].set_axis_off()
            ax[i_row, 0].hist(h_hidden_mean[i, :], bins=40, alpha=0.6, color='blue')
            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            # ax[i_row, 0].set_xticklabels([])
            ax[i_row, 0].set_ylabel(f'Hidden Dim Histogram {i}')

        plt.savefig(plot_dir + '/' + tag + '_hidden' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
        plt.close(fig)
    # ------------------------------------------------------------------------------------------------------------------

    # plot kld values separately
    if plot_klds:
        N_ROWS = kld_values.shape[0] + 1
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                               gridspec_kw={"width_ratios": [60, 1]})
        t_1 = np.linspace(0, 10, kld_values.shape[1])
        t_2 = np.linspace(0, 10, len(signal_1))
        i_row = -1
    # for j in range(Sx.shape[0]):
        for i in range(kld_values.shape[0]):
            i_row += 1
            ax[i_row, 1].set_axis_off()
            ax[i_row, 0].plot(t_1, kld_values[i, :], linewidth=1.5, color="#0C2D57")

            ax2 = ax[i_row, 0].twinx()
            ax3 = ax[i_row, 0].twinx()
            ax2.plot(t_2, signal_1, linewidth=1.5, color="#FE7A36")
            # ax3.plot(t_2, signal, linewidth=2, color="#0D9276")
            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            ax[i_row, 0].set_ylabel(f'KLD-Latent{i}')
        i_row += 1
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(t_1, np.sum(kld_values, axis=0), linewidth=1.5, color="#0C2D57")
        ax3 = ax[i_row, 0].twinx()
        ax3.plot(t_2, signal_1, linewidth=1.5)

        plt.savefig(plot_dir + '/' + tag + '_klds' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
        plt.close(fig)




    # N_ROWS = 2*Sx.shape[0]
    # fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
    #                        gridspec_kw={"width_ratios": [60, 1]}, squeeze=False)
    # i_row = -1
    # for i in range(Sx.shape[0]):
    #     i_row += 1
    #     ax[i_row, 1].set_axis_off()
    #     ax[i_row, 0].plot(Sx[i, :], linewidth=2, color="#003865")
    #     ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    #     ax[i_row, 0].set_xticklabels([])
    #     ax[i_row, 0].set_ylabel('Original Signal')
    #
    #     i_row += 1
    #     ax[i_row, 1].set_axis_off()
    #     ax[i_row, 0].plot(new_sample[i, :], linewidth=1.5, color="#EF5B0C")
    #
    #     ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    #     ax[i_row, 0].set_xticklabels([])
    #     ax[i_row, 0].set_ylabel('New Sample')
    # plt.savefig(plot_dir + '/' + tag + '_new-sample' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=100)
    # plt.close(fig)

    N_ROWS = 2 * Sx.shape[0]


def plot_general_mse(signal=None, plot_order=None, Sx=None, meta=None,
                     Sxr=None, Sxr_std=None, z_latent=None, plot_dir=None, tag='',
                     all_mse=None):

    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})
    i_row = -1
    N_ROWS = (all_mse.shape[0])
    max_mse_value = np.max(all_mse)
    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                           gridspec_kw={"width_ratios": [80, 1]})
    for i in range(N_ROWS):
        i_row += 1
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(all_mse[i, :], linewidth=1.5, marker='.', ms=1)
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_ylim(0, max_mse_value)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel(f'mse coefficient {i}')
        avg_mse = np.mean(all_mse[i, :])
        ax[i_row, 0].set_title(f'Average MSE: {avg_mse:.5f}')
    plt.savefig(plot_dir + '/' + tag + '_mses' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
    plt.close(fig)


def plot_generated_samples(sx, sx_mean, sx_std, input_len, tag='_', plot_dir=None):
    Fs = 4
    log_eps = 1e-3
    N = input_len
    N_ROWS = 3 + (sx.shape[0])
    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})
    i_row = 0

    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                           gridspec_kw={"width_ratios": [80, 1]})
    imgplot = ax[i_row, 0].imshow(sx, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, sx.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Sample')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(sx_mean, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, sx.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Mean')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(sx_std, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, sx.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Standard Deviation')

    for i in range(sx.shape[0]):
        i_row += 1
        ax[i_row, 0].plot(sx_mean[i, :], linewidth=1.25, label="True")
        if sx_std is not None:
            ax[i_row, 0].fill_between(np.arange(len(sx_std[i, :])),
                                      sx_mean[i, :] - sx_std[i, :],
                                      sx_mean[i, :] + sx_std[i, :],
                                      color='blue', alpha=0.1, label='Std dev')
        ax[i_row, 0].legend()
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].set_ylabel(f'Mean and Std of Coefficient {i}')

    cmstr = 'bwr'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    plt.savefig(plot_dir + '/' + tag + '_' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
    plt.close(fig)


def plot_distributions(sx_mean=None, sx_std=None, plot_second_channel=False, plot_sample=False, plot_dir=None,
                       plot_dataset_average=False, sample_sx=None, sample_sx_mean=None, sample_sx_std=None, tag=''):
    Fs = 4
    log_eps = 1e-3
    # N = sx_mean.shape[0]
    # if Sxr is not None:
    N_ROWS = sx_mean.shape[0]
    #     # N_ROWS = 3
    #     N_ROWS = len(plot_order) + 4
    # else:
    #     # N_ROWS = 2
    #     N_ROWS = len(plot_order) + 1
    # if plot_second_channel:
    #     N_ROWS = 5 + (sx.shape[0])
    #     signal_1 = signal[:, 0]
    #     signal_2 = signal[:, 1]
    # else:
    #     N_ROWS = 4 + (Sx.shape[0])
    #     signal_2 = signal
    #     signal_1 = signal
    # t_in = np.arange(0, N) / Fs
    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})

    if plot_dataset_average:
        i_row = -1
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                               gridspec_kw={"width_ratios": [80, 1]})
        for i in range(sx_mean.shape[0]):
            i_row += 1
            ax[i_row, 0].plot(sx_mean[i, :], linewidth=1.5, label=f"st_{i}")
            ax[i_row, 0].fill_between(np.arange(len(sx_mean[i, :])),
                                      sx_mean[i, :] - sx_std[i, :],
                                      sx_mean[i, :] + sx_std[i, :],
                                      color='blue', alpha=0.25, label='Std dev')
            ax[i_row, 0].legend()
            ax[i_row, 1].set_axis_off()
            ax[i_row, 0].set_ylabel(f'St-Coefficient-{i}')
        plt.savefig(plot_dir + '/' + tag + '_dataset' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
        plt.close(fig)

    if plot_sample:
        N_ROWS = sx_mean.shape[0]
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                               gridspec_kw={"width_ratios": [80, 1]})
        i_row = -1
        for i in range(sx_mean.shape[0]):
            i_row += 1
            ax[i_row, 0].plot(sx_mean[i, :], linewidth=1.5, linestyle=(0, (5, 1)), label=f"st_{i}")
            ax[i_row, 0].fill_between(np.arange(len(sx_mean[i, :])),
                                      sx_mean[i, :] - sx_std[i, :],
                                      sx_mean[i, :] + sx_std[i, :],
                                      color='blue', alpha=0.19, label='Std dev')
            ax[i_row, 0].plot(sample_sx_mean[i, :], linewidth=2, color='black')
            ax[i_row, 0].legend()
            ax[i_row, 1].set_axis_off()
            ax[i_row, 0].set_ylabel(f'St-Coefficient-{i}')

        plt.savefig(plot_dir + '/' + tag + '_sample' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
        plt.close(fig)
