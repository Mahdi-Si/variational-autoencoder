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
from vrnn_classifier_GMM_experiment_3 import VRNNGauss, ClassifierBlock, VRNNClassifier
from Variational_AutoEncoder.datasets.custom_datasets import JsonDatasetPreload
from Variational_AutoEncoder.utils.data_utils import plot_scattering_v2, plot_loss_dict
from Variational_AutoEncoder.utils.run_utils import StreamToLogger, setup_logging
from sklearn.manifold import TSNE

from Variational_AutoEncoder.utils.data_utils import plot_scattering_v2, plot_averaged_results, \
    plot_generated_samples, plot_distributions, plot_histogram
import torch.distributions as tdist
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# helper functions you can move them later -----------------------------------------------------------------------------

def calculate_log_likelihood(dec_mean_t_, dec_std_t_, Sx_t_):
    dec_mean_t_ = dec_mean_t_.to(Sx_t_.device)
    dec_std_t_ = dec_std_t_.to(Sx_t_.device)
    pred_dist = tdist.Normal(dec_mean_t_, dec_std_t_)
    log_probs = pred_dist.log_prob(Sx_t_)
    log_likelihoods = log_probs.sum(dim=[1, 2])
    return log_likelihoods.cpu().numpy()

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.00001):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class VrnnGraphModel:
    def __init__(self, config_file_path, batch_size=None, device_id=device,
                 max_length=None, n_input_chan=None, n_output_dim=None, n_aux_labels=None,
                 loss_weights=None):
        super(VrnnGraphModel, self).__init__()
        self.config_file_path = config_file_path
        with open(self.config_file_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        now = datetime.now()
        run_date = now.strftime("%Y-%m-%d--[%H-%M]-")
        self.experiment_tag = config['general_config']['tag']
        # self.output_base_dir = os.path.normpath(config['folders_config']['out_dir_base'])
        self.output_base_dir = os.getcwd() # todo: check this if it works
        self.base_folder = f'{run_date}-{self.experiment_tag}'
        self.train_results_dir = os.path.join(self.output_base_dir, self.base_folder, 'train_results')
        self.test_results_dir = os.path.join(self.output_base_dir, self.base_folder, 'test_results')
        self.model_checkpoint_dir = os.path.join(self.output_base_dir, self.base_folder, 'model_checkpoints')
        self.aux_dir = os.path.join(self.output_base_dir, self.base_folder, 'aux_test_HIE')
        self.tensorboard_dir = os.path.join(self.output_base_dir, self.base_folder, 'tensorboard_log')

        self.log_file = None
        self.logger = None

        # print yaml file properly -------------------------------------------------------------------------------------
        print(yaml.dump(config, sort_keys=False, default_flow_style=False))
        print('==' * 50)
        self.stat_path = os.path.normpath(config['dataset_config']['stat_path'])

        self.plot_every_epoch = config['general_config']['plot_frequency']
        self.previous_check_point = config['general_config']['checkpoint_path']

        self.raw_input_size = config['model_config']['VAE_model']['raw_input_size']
        self.input_size = config['model_config']['VAE_model']['input_size']
        self.input_dim = config['model_config']['VAE_model']['input_dim']
        self.latent_dim = config['model_config']['VAE_model']['latent_size']
        self.num_layers = config['model_config']['VAE_model']['num_RNN_layers']
        self.rnn_hidden_dim = config['model_config']['VAE_model']['RNN_hidden_dim']
        self.epochs_num = config['general_config']['epochs']
        self.lr = config['general_config']['lr']
        self.lr_milestones = config['general_config']['lr_milestone']
        self.kld_beta_ = float(config['model_config']['VAE_model']['kld_beta'])
        self.vrnn_checkpoint_path = config['model_config']['vrnn_checkpoint']
        self.freeze_vrnn = config['model_config']['VAE_model']['freeze_vrnn']

        self.test_checkpoint_path = None

        # hyperparameters
        x_dim = self.input_dim
        h_dim = self.rnn_hidden_dim
        z_dim = self.latent_dim
        n_layers = self.num_layers
        n_epochs = self.epochs_num
        self.clip = 10
        learning_rate = self.lr
        plt.ion()

        self.log_stat = None
        self.model = None
        self.vrnn_model = None
        self.classifier = None
        self.early_stopping = EarlyStopping(patience=20, verbose=True, delta=0.008)

    def setup_config(self):
        torch.manual_seed(42)
        np.random.seed(42)
        folders_list = [self.output_base_dir, self.train_results_dir, self.test_results_dir, self.model_checkpoint_dir,
                        self.aux_dir, self.tensorboard_dir]
        for folder in folders_list:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # setting up the logging -------------------------------------------------------------------------------------------
        self.log_file = os.path.join(self.train_results_dir, 'log.txt')
        self.logger = setup_logging(log_file_setup=self.log_file)
        sys.stdout = StreamToLogger(self.logger, logging.INFO)

        with open(self.stat_path, 'rb') as f:
            x_mean = np.load(f)
            x_std = np.load(f)
        self.log_stat = (x_mean, x_std)

    def create_model(self):
        self.setup_config()
        self.vrnn_model = VRNNGauss(input_dim=self.input_dim, input_size=self.raw_input_size,
                                    h_dim=self.rnn_hidden_dim, z_dim=self.latent_dim, n_layers=self.num_layers,
                                    device=device, log_stat=self.log_stat, bias=False).to(device)
        self.classifier = ClassifierBlock(conv_in_channels=9, conv_out_channels=9, conv_kernel_size=(1, 1),
                                          conv_depth_multiplier=1, conv_activation=nn.ReLU(), lstm_input_dim=9,
                                          lstm_h=[14, 19, 24, 32], lstm_bidirectional=False).to(device)
        self.model = VRNNClassifier(self.vrnn_model, self.classifier)

        self.model = self.model.to(device)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Trainable params: {trainable_params}')
        print('==' * 50)
        return self

    # def load_pretrained_vrnn(self, checkpoint_path):
    #     # This will load the pretrained VRNN without classification model.
    #
    #     # model_dict = self.model.vrnn_model.state_dict()  # Get the state dictionary of the model
    #     pretrained_dict = torch.load(checkpoint_path)  # Load the checkpoint
    #     # Filter out unnecessary keys and update the state dictionary
    #     # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     # model_dict.update(pretrained_dict)
    #     self.model
    #     self.model.vrnn_model.load_state_dict(model_dict)  # Load the updated state dictionary into the model
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth.tar'):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss,
        }, filename)

    @staticmethod
    def load_checkpoint(model, previous_check_point, optimizer):
        checkpoint = torch.load(previous_check_point)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{previous_check_point}' (epoch {checkpoint['epoch']})")

    # todo: look into how to load the optimizer checkpoint properly

    @staticmethod
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False

    def train_vrnn_model(self, epoch_train=None, plot_dir=None, tag='', vrnn_train_loader=None, optimizer=None):
        for param_group in optimizer.param_groups:
            current_learning_rate = param_group['lr']
            print(f'Learning Rate; {current_learning_rate}')
        train_loss_epoch = 0
        nll_loss_epoch = 0
        kld_loss_epoch = 0
        plt.close('all')
        vrnn_train_loader_tqdm = tqdm(enumerate(vrnn_train_loader), total=len(vrnn_train_loader),
                                      desc=f'Epoch {epoch_train}')
        self.model.vrnn_model.train()
        for batch_idx, train_data in vrnn_train_loader_tqdm:
            data = train_data[0]
            data = data.squeeze(-1).to(device)
            results = self.model.vrnn_model(data)
            loss = (self.kld_beta_ * results.kld_loss) + results.nll_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            kld_loss_epoch += self.kld_beta_ * results.kld_loss.item()
            nll_loss_epoch += results.nll_loss.item()
            train_loss_epoch += loss.item()

            nn.utils.clip_grad_norm_(self.model.vrnn_model.parameters(), self.clip)
            message = (f'Train Epoch: {epoch_train} [{batch_idx * len(data)}/{len(train_loader)} '
                       f'({100. * batch_idx / len(train_loader):.0f}%)] | '
                       f'-KLD Loss: {results.kld_loss.item():.5f} - \
                        Weighted KLD Loss: {self.kld_beta_ * results.kld_loss:.5f} | '
                       f'-Reconstruction Loss: {results.rec_loss.item():.5f}')
            print(message)

            if epoch_train % self.plot_every_epoch == 0:
                if batch_idx % 100 == 0:
                    signal_ = data[0]
                    sx_ = results.Sx.permute(1, 2, 0)[0]
                    z_latent_ = results.z_latent[0]
                    dec_mean_ = results.decoder_mean[0]
                    # signal = signal_.squeeze(0).permute(1, 0).detach().cpu().numpy()  # for two channels
                    signal = signal_.detach().cpu().numpy()  # for one channels
                    plot_scattering_v2(signal=signal,
                                       Sx=sx_.detach().cpu().numpy(),
                                       meta=None,
                                       plot_second_channel=False,
                                       Sxr=dec_mean_.detach().cpu().numpy(),
                                       z_latent=z_latent_.detach().cpu().numpy(),
                                       plot_dir=plot_dir, tag=f'_epoch{epoch_train}_batch_{batch_idx}_train')
        train_loss_tl_avg = train_loss_epoch / len(vrnn_train_loader.dataset)
        nll_loss_tl_avg = nll_loss_epoch / len(vrnn_train_loader.dataset)
        kld_loss_tl_avg = kld_loss_epoch / len(vrnn_train_loader.dataset)
        print(f'Train Loss Mean: {train_loss_tl_avg}')
        return train_loss_tl_avg, nll_loss_tl_avg, kld_loss_tl_avg

    def validate_vrnn_model(self, epoch_validation=None, plot_dir=None, tag='', vrnn_validation_loader=None):
        validation_loss_epoch = 0
        nll_valid_loss_epoch = 0
        kld_valid_loss_epoch = 0
        plt.close('all')
        vrnn_validation_loader_tqdm = tqdm(enumerate(vrnn_validation_loader), total=len(vrnn_validation_loader),
                                           desc=f'Epoch {epoch_validation}')
        self.model.vrnn_model.eval()
        with torch.no_grad():
            for batch_idx, validation_data in vrnn_validation_loader_tqdm:
                data = validation_data[0]
                data = data.squeeze(-1).to(device)
                results = self.model.vrnn_model(data)
                loss = (self.kld_beta_ * results.kld_loss) + results.nll_loss
                kld_valid_loss_epoch += self.kld_beta_ * results.kld_loss.item()
                nll_valid_loss_epoch += results.nll_loss.item()
                validation_loss_epoch += loss.item()
                if epoch_validation % self.plot_every_epoch == 0:
                    if batch_idx % 100 == 0:
                        signal_ = data[0]
                        sx_ = results.Sx.permute(1, 2, 0)[0]
                        z_latent_ = results.z_latent[0]
                        dec_mean_ = results.decoder_mean[0]
                        # signal = signal_.squeeze(0).permute(1, 0).detach().cpu().numpy()  # for two channels
                        signal = signal_.detach().cpu().numpy()  # for one channels
                        plot_scattering_v2(signal=signal,
                                           Sx=sx_.detach().cpu().numpy(),
                                           meta=None,
                                           plot_second_channel=False,
                                           Sxr=dec_mean_.detach().cpu().numpy(),
                                           z_latent=z_latent_.detach().cpu().numpy(),
                                           plot_dir=plot_dir, tag=f'_epoch{epoch_validation}_batch_{batch_idx}_train')
        validating_loss_tl_avg = validation_loss_epoch / len(vrnn_validation_loader.dataset)
        nll_valid_loss_tl_avg = nll_valid_loss_epoch / len(vrnn_validation_loader.dataset)
        kld_loss_tl_avg = kld_valid_loss_epoch / len(vrnn_validation_loader.dataset)
        print(f'Validation Loss Mean: {validating_loss_tl_avg}')
        return validating_loss_tl_avg, nll_valid_loss_tl_avg, kld_loss_tl_avg

    def do_train_vrnn_model(self, vrnn_train_dataset, vrnn_validation_dataset):
        """
        This method only trains the VRNN model. It can be used as pretraining the vrnn part. It will save a checkpoint
        of the model at lowest validation loss.
        :param vrnn_train_dataset: (torch.utils.data.dataset.DataLoader) The training dataloader.
        :param vrnn_validation_dataset: (torch.utils.data.dataset.Dataset) The validation dataloader.
        :return: None
        """
        optimizer = torch.optim.Adam(self.vrnn_model.parameters(), lr=self.lr)
        schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.lr_milestones)

        if self.vrnn_checkpoint_path is not None:
            print(f"Loading checkpoint: {self.vrnn_checkpoint_path}")
            checkpoint = torch.load(self.vrnn_checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            self.model.vrnn_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{self.vrnn_checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            start_epoch = 1

        train_loss_list = []
        train_nll_loss_list = []
        train_kld_loss_list = []

        test_loss_list = []
        test_nll_loss_list = []
        test_kld_loss_list = []

        for epoch in tqdm(range(start_epoch, self.epochs_num + 1), desc=f'Epoch '):
            train_loss, train_nll_loss, train_kld_loss = self.train_vrnn_model(epoch_train=epoch,
                                                                               plot_dir=self.train_results_dir,
                                                                               vrnn_train_loader=vrnn_train_dataset,
                                                                               optimizer=optimizer)
            schedular.step()
            train_loss_list.append(train_loss)
            train_nll_loss_list.append(train_nll_loss)
            train_kld_loss_list.append(train_kld_loss)

            validation_loss, validation_nll_loss, validation_kld_loss = self.validate_vrnn_model(
                epoch_validation=epoch,
                plot_dir=self.test_results_dir,
                vrnn_validation_loader=vrnn_validation_dataset,
            )

            test_loss_list.append(validation_loss)
            test_nll_loss_list.append(validation_nll_loss)
            test_kld_loss_list.append(validation_kld_loss)

            validation_loss_min = min(test_loss_list)

            loss_dict = {
                'train_loss': train_loss_list,
                'test_loss': test_loss_list,
                'train_nll_loss': train_nll_loss_list,
                'test_nll_loss': test_nll_loss_list,
                'train_kld_loss': train_kld_loss_list,
                'test_kld_loss': test_kld_loss_list,
            }

            if len(test_loss_list) > 0:
                if validation_loss <= validation_loss_min:
                    checkpoint_name = f'VRNN-{epoch}.pth'
                    vrnn_checkpoint_path = os.path.join(self.model_checkpoint_dir, checkpoint_name)

                    self.save_checkpoint(model=self.model.vrnn_model,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         loss=loss_dict,
                                         filename=vrnn_checkpoint_path)

            loss_path = os.path.join(self.train_results_dir, f'loss_dict.pkl')
            if epoch % self.plot_every_epoch == 0:
                with open(loss_path, 'wb') as file:
                    pickle.dump(loss_dict, file)
                plot_loss_dict(loss_dict=loss_dict,
                               epoch_num=epoch,
                               plot_dir=self.train_results_dir)

    # new methods to add start -----------------------------------------------------------------------------------------
    def plot_vrnn_tests(self, vrnn_plot_tests_data_loader, input_dim_t,
                        modify_h=None, modify_z=None, base_dir=None, channel_num=1):
        self.model.vrnn_model.modify_z = modify_z
        self.model.vrnn_model.modify_h = modify_h
        self.model.vrnn_model.to(device)
        self.model.vrnn_model.eval()
        mse_all_data = torch.empty((0, input_dim_t)).to(device)
        log_likelihood_all_data = []
        all_st = []
        with torch.no_grad():
            for j, complete_batched_data_t in tqdm(enumerate(vrnn_plot_tests_data_loader),
                                                   total=len(vrnn_plot_tests_data_loader)):
                batched_data_t = complete_batched_data_t[0]
                # guids = complete_batched_data_t[1]
                batched_data_t = batched_data_t.to(device)  # (batch_size, signal_len)
                results_t = self.model.vrnn_model(batched_data_t)
                z_latent_t_ = results_t.z_latent  # (batch_size, latent_dim, 150)
                h_hidden_t_ = results_t.hidden_states  # (hidden_layers, batch_size, input_len, h_dim)
                if h_hidden_t_.dim() == 4:
                    h_hidden_t__ = h_hidden_t_[-1].permute(0, 2, 1)
                else:
                    h_hidden_t__ = h_hidden_t_.permute(0, 2, 1)
                dec_mean_t_ = results_t.decoder_mean  # (batch_size, input_dim, input_size)
                dec_std_t_ = torch.sqrt(torch.exp(results_t.decoder_std))
                Sx_t_ = results_t.Sx.permute(1, 2, 0)  # (batch_size, input_dim, 150)
                enc_mean_t_ = results_t.encoder_mean  # (batch_size, input_dim, 150)
                enc_std_t_ = torch.sqrt(torch.exp(results_t.encoder_std))
                kld_values_t_ = results_t.kld_values

                mse_per_coefficients = torch.sum(((Sx_t_ - dec_mean_t_) ** 2), dim=2) / Sx_t_.size(-1)
                mse_all_data = torch.cat((mse_all_data, mse_per_coefficients), dim=0)
                log_likelihoods = calculate_log_likelihood(dec_mean_t_, dec_std_t_, Sx_t_)
                log_likelihood_all_data.extend(log_likelihoods)
                all_st.append(Sx_t_)
                save_dir = os.path.join(base_dir, 'Complete vrnn testing')
                os.makedirs(save_dir, exist_ok=True)
                signal_channel_dim = Sx_t_.shape[1]
                signal_len = Sx_t_.shape[2]
                for signal_index in range(Sx_t_.shape[0]):
                    save_dir_signal = save_dir
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
                                          plot_dir=save_dir_signal, tag=f'-{signal_index}')
                    plot_scattering_v2(signal=signal_c,
                                       plot_second_channel=two_channel_flag,
                                       Sx=sx_selected.detach().cpu().numpy(), meta=None,
                                       Sxr=dec_mean_t_[signal_index].detach().cpu().numpy(),
                                       Sxr_std=dec_std_t_[signal_index].detach().cpu().numpy(),
                                       z_latent=enc_mean_t_[signal_index].detach().cpu().numpy(),
                                       plot_dir=save_dir_signal, tag=f'-{signal_index}')

    def vrnn_mse_test(self, vrnn_mse_test_dataloader,  input_dim_t, modify_h=None, modify_z=None, base_dir=None,
                      tag="_"):
        self.model.vrnn_model.modify_z = modify_z
        self.model.vrnn_model.modify_h = modify_h
        self.model.vrnn_model.to(device)
        self.model.vrnn_model.eval()
        mse_all_data = torch.empty((0, input_dim_t)).to(device)
        epoch_data_collected = []
        log_likelihood_all_data = []
        all_st = []
        with torch.no_grad():
            for j, complete_batched_data_t in tqdm(enumerate(vrnn_mse_test_dataloader),
                                                   total=len(vrnn_mse_test_dataloader)):
                batched_data_t = complete_batched_data_t[0]
                batched_data_t = batched_data_t.to(device)  # (batch_size, signal_len)
                results_t =  self.model.vrnn_model(batched_data_t)
                dec_mean_t_ = results_t.decoder_mean  # (batch_size, input_dim, input_size)
                dec_std_t_ = torch.sqrt(torch.exp(results_t.decoder_std))
                Sx_t_ = results_t.Sx.permute(1, 2, 0)  # (batch_size, input_dim, 150)
                mse_per_coefficients = torch.sum(((Sx_t_ - dec_mean_t_) ** 2), dim=2) / Sx_t_.size(-1)
                mse_all_data = torch.cat((mse_all_data, mse_per_coefficients), dim=0)
                log_likelihoods = calculate_log_likelihood(dec_mean_t_, dec_std_t_, Sx_t_)
                log_likelihood_all_data.extend(log_likelihoods)
                all_st.append(Sx_t_)
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
        plot_histogram(data=mse_all_data_averaged.detach().cpu().numpy() / 150,
                       single_channel=True,
                       bins=160, save_dir=save_dir_hist, tag='mse-all_dist')
        plot_histogram(data=mse_all_data.detach().cpu().numpy(),
                       single_channel=False,
                       bins=160, save_dir=save_dir_hist, tag='mse-all-data-per')
        return all_st_tensor
    def do_test_vrnn_model(self, vrnn_test_dataset):
        self.plot_vrnn_tests(vrnn_plot_tests_data_loader=vrnn_test_dataset, input_dim_t=self.input_dim,
                             modify_h=None, modify_z=None, base_dir=self.test_results_dir, channel_num=1)
        self.vrnn_mse_test(vrnn_mse_test_dataloader=vrnn_test_dataset, input_dim_t=self.input_dim,
                           modify_h=None, modify_z=None, base_dir=self.test_results_dir, tag='vrnn_mse')


    # new methods to add end -----------------------------------------------------------------------------------------
    def train(self, epoch_train=None, kld_beta=1.1, plot_dir=None, tag='', train_loader_classifier=None,
              optimizer=None, plot_every_epoch=None, loss_fn_classifier=None):
        for param_group in optimizer.param_groups:
            current_learning_rate = param_group['lr']
            print(f'Learning Rate; {current_learning_rate}')
        reconstruction_loss_epoch = 0
        kld_loss_epoch = 0
        nll_loss_epoch = 0
        total_correct = 0
        total_samples = 0
        total_loss = 0
        plt.close('all')
        train_loader_tqdm = tqdm(enumerate(train_loader_classifier), total=len(train_loader_classifier),
                                 desc=f"Epoch {epoch_train}")
        self.model.train()
        for batch_idx, train_data in train_loader_tqdm:
            data = train_data[0]
            data = data.squeeze(-1).to(device)  # shape: (batch, input_length, indput_dim) (10, 4800, 1)
            results, classifier_output = self.model(data)

            sample_weights = train_data[1].to(device)
            true_targets = (train_data[2]).to(device)
            batch_size_, original_length, num_classes = true_targets.shape
            # true_targets = true_targets.view(batch_size_, original_length // 16, 16, num_classes).sum(dim=2).argmax(dim=2)
            # sample_weights = sample_weights.view(batch_size_, original_length // 16, 16)
            true_targets = true_targets.argmax(dim=-1)

            true_targets_ = true_targets.view(-1)
            classifier_output_ = classifier_output.view(-1, num_classes)
            sample_weights = sample_weights.view(-1)

            none_zero_mask = (true_targets_ != 0)

            filtered_true_targets = true_targets_[none_zero_mask]
            filtered_classifier_output = classifier_output_[none_zero_mask]
            filtered_sample_weights = sample_weights[none_zero_mask]

            # true_targets = torch.argmax(true_targets, dim=1)
            # valid_mask = (true_targets != 0)
            # valid_mask_expanded = valid_mask.unsqueeze(-1).expand(-1, -1, 3)
            # masked_true_targets = true_targets[valid_mask]
            # masked_classifier_output = classifier_output[valid_mask_expanded].view(-1, 3)

            optimizer.zero_grad()
            classification_loss = loss_fn_classifier(filtered_classifier_output, filtered_true_targets)
            classification_loss = filtered_sample_weights * classification_loss
            loss_vrnn = (kld_beta * results.kld_loss) + results.nll_loss
            loss = classification_loss.mean()
            loss.backward()
            optimizer.step()
            kld_loss_epoch += kld_beta * results.kld_loss.item()
            # nll_loss_epoch += results.nll_loss.item()
            total_loss += loss.item()
            reconstruction_loss_epoch += results.rec_loss.item()

            predicted_labels = torch.argmax(classifier_output, dim=2)
            correct_predictions = (predicted_labels == true_targets).sum().item()

            total_correct += correct_predictions
            total_samples += filtered_true_targets.size(0)

            # grad norm clipping, only in pytorch version >= 1.10
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            if epoch_train % plot_every_epoch == 0:
                if batch_idx % 100 == 0:
                    signal_ = data[0]
                    sx_ = results.Sx.permute(1, 2, 0)[0]
                    z_latent_ = results.z_latent[0]
                    dec_mean_ = results.decoder_mean[0]
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
        train_loss_tl_avg = total_loss / len(train_loader_classifier)
        reconstruction_loss_avg = reconstruction_loss_epoch / len(train_loader_classifier.dataset)
        kld_loss_avg = kld_loss_epoch / len(train_loader_classifier.dataset)
        accuracy = total_correct / total_samples

        print(f'Epoch: {epoch_train} - Average Train Loss Per Batch: {train_loss_tl_avg} \n Train accuracy is: {accuracy}')
        return train_loss_tl_avg, accuracy

    def validate_model(self, epoch_validation=None, validation_loader=None,
                       loss_fn_classifier=None, kld_beta=1.1, plot_dir=None, tag='',
                       plot_every_epoch=None):
        reconstruction_loss_epoch = 0
        kld_loss_epoch = 0
        nll_loss_epoch = 0
        total_correct = 0
        total_samples = 0
        total_loss = 0
        plt.close('all')
        validation_loader_tqdm = tqdm(enumerate(validation_loader), total=len(validation_loader))
        self.model.eval()
        # [X] done -ttodo: change the data from dataloader based on the mimo_trainer
        with torch.no_grad():
            for batch_idx, train_data in validation_loader_tqdm:
                data = train_data[0]
                data = data.squeeze(-1).to(device)  # shape: (batch, input_length, indput_dim) (10, 4800, 1)
                results, classifier_output = self.model(data)

                sample_weights = train_data[1].to(device)
                true_targets = (train_data[2]).to(device)
                batch_size_, original_length, num_classes = true_targets.shape
                true_targets = true_targets.argmax(dim=-1)

                true_targets_ = true_targets.view(-1)
                classifier_output_ = classifier_output.view(-1, num_classes)
                sample_weights = sample_weights.view(-1)
                none_zero_mask = (true_targets_ != 0)

                filtered_true_targets = true_targets_[none_zero_mask]
                filtered_classifier_output = classifier_output_[none_zero_mask]
                filtered_sample_weights = sample_weights[none_zero_mask]

                classification_loss = loss_fn_classifier(filtered_classifier_output, filtered_true_targets)
                classification_loss = filtered_sample_weights * classification_loss
                loss_vrnn = (kld_beta * results.kld_loss) + results.nll_loss
                loss = classification_loss.mean()
                kld_loss_epoch += kld_beta * results.kld_loss.item()
                # nll_loss_epoch += results.nll_loss.item()
                total_loss += loss.item()
                reconstruction_loss_epoch += results.rec_loss.item()

                predicted_labels = torch.argmax(classifier_output, dim=2)
                correct_predictions = (predicted_labels == true_targets).sum().item()

                total_correct += correct_predictions
                total_samples += filtered_true_targets.size(0)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        validation_loss_avg = total_loss / len(validation_loader)
        reconstruction_loss_avg = reconstruction_loss_epoch / len(validation_loader.dataset)
        kld_loss_avg = kld_loss_epoch / len(validation_loader.dataset)
        accuracy = total_correct / total_samples

        print(f'Epoch: {epoch_validation} - Average Validation Loss Per Batch: {validation_loss_avg} \n Validation accuracy is: {accuracy}')
        return validation_loss_avg, accuracy

    # todo: figure out a unified way of saving the model checkpoint  Handle the best possible way to save checkpoints, because you are doing it in two places
    def do_train_with_dataset(self, train_dataset, validation_dataset, tag='', weights_filename=None):
        optimizer = torch.optim.Adam(list(self.model.vrnn_model.parameters()) +
                                     list(self.model.classifier_model.parameters()),
                                     lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.lr_milestones)
        # if self.previous_check_point is not None:
        #     print(f"Loading checkpoint '{self.previous_check_point}'")
        #     checkpoint = torch.load(self.previous_check_point)
        #     start_epoch = checkpoint['epoch'] + 1
        #     self.model.load_state_dict(checkpoint['state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     print(f"Loaded checkpoint '{self.previous_check_point}' (epoch {checkpoint['epoch']})")
        # else:
        #     start_epoch = 1

        if self.vrnn_checkpoint_path is not None:
            self.load_checkpoint(model=self.model.vrnn_model,
                                 previous_check_point=self.vrnn_checkpoint_path,
                                 optimizer=None)
        if self.freeze_vrnn:
            self.freeze_model(self.vrnn_model)
        self.model = self.model.to(device)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Trainable params: {trainable_params}')
        print('==' * 50)

        start_epoch = 1
        train_loss_list = []
        train_accuracy_list = []

        validation_loss_list = []
        validation_accuracy_list = []

        for epoch in tqdm(range(start_epoch, self.epochs_num + 1), desc='Epoch:'):
            train_loss, train_accuracy = self.train(epoch_train=epoch,
                                                    plot_dir=self.train_results_dir,
                                                    plot_every_epoch=self.plot_every_epoch,
                                                    train_loader=train_dataset,
                                                    kld_beta=self.kld_beta_,
                                                    optimizer=optimizer,
                                                    loss_fn_classifier=loss_fn)
            schedular.step()
            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)
            validation_loss, validation_accuracy = self.validate_model(epoch_validation=epoch,
                                                                       validation_loader=validation_dataset,
                                                                       loss_fn_classifier=loss_fn,
                                                                       kld_beta=1.1,
                                                                       plot_dir=None, tag='',
                                                                       plot_every_epoch=self.plot_every_epoch)
            validation_loss_list.append(validation_loss)
            validation_accuracy_list.append(validation_accuracy)

            if len(train_loss_list) > 0:
                if validation_loss <= min(validation_loss_list):
                    self.save_checkpoint(model=self.model.vrnn_model, optimizer=optimizer,
                                         epoch=epoch, loss=validation_loss,
                                         filename=f'vrnn_checkpoint_{epoch}.pth')
                    self.save_checkpoint(model=self.model.classifier_model, optimizer=optimizer,
                                         epoch=epoch, loss=validation_loss,
                                         filename=f'classifier_checkpoint_{epoch}.pth')
                    full_checkpoint_name = f'vrnn-complete-model-{epoch}.pth'
                    self.test_checkpoint_path = os.path.join(self.train_results_dir, full_checkpoint_name)
                    self.save_checkpoint(model=self.model, optimizer=optimizer,
                                         epoch=epoch, loss=validation_loss,
                                         filename=self.test_checkpoint_path)
                    weights_parent_dir = os.path.dirname(weights_filename)
                    if not os.path.exists(weights_parent_dir):
                        os.makedirs(weights_parent_dir)
                    if os.path.exists(weights_filename):
                        os.remove(weights_filename)
                    self.save_checkpoint(model=self.model,
                                         epoch=epoch, loss=validation_loss, optimizer=optimizer,
                                         filename=weights_filename)

            schedular.step()

            loss_dict = {'train_loss': train_loss_list,
                         'test_loss': validation_loss_list,
                         }

            loss_path = os.path.join(self.train_results_dir, 'loss_dict.pkl')
            if epoch % self.plot_every_epoch == 0:
                with open(loss_path, 'wb') as file:
                    pickle.dump(loss_dict, file)
                plot_loss_dict(loss_dict=loss_dict, epoch_num=epoch, plot_dir=self.train_results_dir)

            self.early_stopping(validation_loss)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break


        history_dict = {
            'loss': train_loss_list,
            'accuracy': train_accuracy_list,
            'val_loss': validation_loss_list,
            'val_accuracy': validation_accuracy_list
        }




        return history_dict
        # [X] todo You need to return the validation loss as well

    def predict_dataset(self, dataloader=None, model=None, model_name='default'):
        if self.model is None:
            self.model = model
        # todo: how to load the checkpoint at inference when you want?
        self.load_checkpoint(self.model, self.test_checkpoint_path, optimizer=None)
        self.model.eval()
        all_predictions = []
        data_loader_tqdm = tqdm(enumerate(dataloader), total=len(dataloader))
        with torch.no_grad():
            for batch_idx, batched_data in data_loader_tqdm:
                data = batched_data[0]
                data = data.squeeze(-1).to(device)
                _, logits = self.model(data)
                predictions_ = torch.softmax(logits, dim=-1)
                all_predictions.append(predictions_.cpu().detach().numpy())
        all_predictions = np.concatenate(all_predictions, axis=0)
        return all_predictions

# todo: folder for different folds in vrnn graph model


if __name__ == '__main__':
    config_file_path = r'config_arguments.yaml'
    graph_model = VrnnGraphModel(config_file_path=config_file_path)
    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    dataset_dir = os.path.normpath(config['dataset_config']['dataset_dir'])
    aux_dataset_hie_dir = os.path.normpath(config['dataset_config']['aux_dataset_dir'])
    batch_size = config['general_config']['batch_size']['train']
    fhr_healthy_dataset = JsonDatasetPreload(dataset_dir)
    dataset_size = len(fhr_healthy_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    print(f'Train size: {train_size} \n Test size: {test_size}')
    graph_model.create_model()
    train_dataset, test_dataset = random_split(fhr_healthy_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    print(f'Train size: {len(train_dataset)} \n Test size: {len(test_dataset)}')
    print('==' * 50)
    graph_model.do_train_vrnn_model(train_loader, test_loader)
    # history_ = graph_model.do_train_with_dataset(train_loader, test_loader)
    # predictions = graph_model.predict_dataset(dataloader=train_loader)
    # print('done')
