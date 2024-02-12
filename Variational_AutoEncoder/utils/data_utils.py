import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import torch
import torch.nn as nn
import torchvision
import numpy as np
from kymatio.torch import Scattering1D
import pickle
from scipy.signal import decimate



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


def plot_scattering(signal=None, Sx=None, meta=None, do_plot_rec=False, Sxr=None, plot_dir=None, tag=''):
    '''
    Get fhr and scattering transfrom of it (selected orders) and plot them
    you need the meta of the model as well
    :param signal:
    :param x_transformed:
    :return:
    '''
    signal = signal.cpu().detach().numpy()
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
    N = 4800
    if do_plot_rec:
        N_ROWS = 3
    else:
        N_ROWS = 2
    t = np.linspace(0, 20 * 60, len(signal))

    t_in = np.arange(0, N) / Fs

    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 4, 'axes.titlesize': 4, 'axes.labelsize': 4})
    i_row = 0

    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(4, 6),
                           gridspec_kw={"width_ratios": [40, 1]})

    ax[i_row, 0].plot(t_in, signal, linewidth=0.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(np.log(Sx + log_eps), aspect='auto',
                                  extent=[0, N / Fs, Sx.shape[0], 0])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Order 1')
    fig.colorbar(imgplot, cax=ax[i_row, 1])

    if do_plot_rec:
        i_row += 1
        Sxr = Sxr.cpu().detach().numpy()
        imgplot = ax[i_row, 0].imshow(np.log(Sxr + log_eps), aspect='auto',
                                      extent=[0, N / Fs, Sxr.shape[0], 0])
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('Order 1')
        fig.colorbar(imgplot, cax=ax[i_row, 1])



    cmstr = 'bwr'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    plt.savefig(plot_dir + '/' + tag + '_' + '_st.png', bbox_inches='tight', orientation='landscape')


def plot_original_reconstructed(original_x, reconstructed_x, plot_dir=None, tag=''):
    import matplotlib as mpl
    # Set the font globally to Times New Roman, size 18
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 14

    fig, ax = plt.subplots(3, 1, figsize=(18, 7))
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
