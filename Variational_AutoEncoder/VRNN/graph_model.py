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
import torch.nn.functional as F
import numpy as np
from vrnn_gauss_experiment_6 import VRNNGauss, ClassifierBlock, VRNNClassifier
from Variational_AutoEncoder.datasets.custom_datasets import JsonDatasetPreload, FhrUpPreload
from Variational_AutoEncoder.utils.data_utils import plot_scattering_v2, plot_loss_dict
from Variational_AutoEncoder.utils.run_utils import log_resource_usage, StreamToLogger, setup_logging
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphModel:
    def __init__(self, config_file_path, batch_size=None, device_id=device,
                 max_length=None, n_input_chan=None, n_output_dim=None, n_aux_labels=None,
                 loss_weights=None):
        super(GraphModel, self).__init__()
        self.config_file_path = config_file_path
        with open(self.config_file_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        now = datetime.now()
        run_date = now.strftime("%Y-%m-%d--[%H-%M]-")
        self.experiment_tag = config['general_config']['tag']
        self.output_base_dir = os.path.normpath(config['folders_config']['out_dir_base'])
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

        self.previous_check_point = config['general_config']['checkpoint_path']

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
        self.vrnn_model = VRNNGauss(input_dim=self.input_dim, input_size=self.raw_input_size,
                                    h_dim=self.rnn_hidden_dim, z_dim=self.latent_dim, n_layers=self.num_layers,
                                    device=device, log_stat=self.log_stat, bias=False).to(device)
        self.freeze_model(self.vrnn_model)
        self.classifier = ClassifierBlock(conv_in_channels=3, conv_out_channels=3, conv_kernel_size=(1, 1),
                                          conv_depth_multiplier=1, conv_activation=nn.ReLU(), lstm_input_dim=3,
                                          lstm_h=[6, 7], lstm_bidirectional=False).to(device)
        self.model = VRNNClassifier(self.vrnn_model, self.classifier)
        self.model = self.model.to(device)  # todo: is this the write place for this?
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Trainable params: {trainable_params}')
        print('==' * 50)
        return self

    @staticmethod
    def load_pretrained_vrnn(model, checkpoint_path):
        # This will load the pretrained VRNN without classification model.
        model_dict = model.state_dict()  # Get the state dictionary of the model
        pretrained_dict = torch.load(checkpoint_path)  # Load the checkpoint
        # Filter out unnecessary keys and update the state dictionary
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)  # Load the updated state dictionary into the model
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth.tar'):
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, filename)

    @staticmethod
    def load_checkpoint(model, previous_check_point, optimizer):
        checkpoint = torch.load(previous_check_point)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{previous_check_point}' (epoch {checkpoint['epoch']})")

    # todo: look into how to load the optimizer checkpoint properly

    @staticmethod
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self, epoch_train=None, kld_beta=1.1, plot_dir=None, tag='', train_loader=None,
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
        train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_train}")
        if self.vrnn_checkpoint_path is not None:
            self.load_pretrained_vrnn(model=self.vrnn_model, checkpoint_path=self.vrnn_checkpoint_path)
        if self.freeze_vrnn:
            self.freeze_model(self.model.vrnn_model)
        self.model.train()
        # todo: change the data from dataloader based on the mimo_trainer
        for batch_idx, train_data in train_loader_tqdm:
            data = train_data[0]
            data = data.to(device)
            true_targets = (train_data[3]).to(device)
            valid_mask = (true_targets != 0)
            valid_mask_expanded = valid_mask.unsqueeze(-1).expand(-1, -1, 3)
            masked_true_targets = true_targets[valid_mask]
            optimizer.zero_grad()
            results, classifier_output = self.model(data)
            masked_classifier_output = classifier_output[valid_mask_expanded].view(-1, 3)
            classification_loss = loss_fn_classifier(masked_classifier_output, masked_true_targets)
            loss_vrnn = (kld_beta * results.kld_loss) + results.nll_loss
            loss = classification_loss
            loss.backward()
            optimizer.step()
            kld_loss_epoch += kld_beta * results.kld_loss.item()
            # nll_loss_epoch += results.nll_loss.item()
            total_loss += loss.item()
            reconstruction_loss_epoch += results.rec_loss.item()

            predicted_labels = torch.argmax(masked_classifier_output, dim=1)
            correct_predictions = (predicted_labels == masked_true_targets).sum().item()

            total_correct += correct_predictions
            total_samples += masked_true_targets.size(0)

            # grad norm clipping, only in pytorch version >= 1.10
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
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
                    # plot_scattering_v2(signal=signal,
                    #                    Sx=sx_.detach().cpu().numpy(),
                    #                    meta=None,
                    #                    plot_second_channel=False,
                    #                    Sxr=dec_mean_.detach().cpu().numpy(),
                    #                    z_latent=z_latent_.detach().cpu().numpy(),
                    #                    plot_dir=plot_dir, tag=f'_epoch{epoch_train}_batch_{batch_idx}_train')

        # print(f'Train Loop train loss is ====> {train_loss_tl}')
        # print('====> Epoch: {} Average loss: {:.4f}'.format(
        #     epoch, train_loss_tl / len(train_loader.dataset)))
        train_loss_tl_avg = total_loss / len(train_loader)
        reconstruction_loss_avg = reconstruction_loss_epoch / len(train_loader.dataset)
        kld_loss_avg = kld_loss_epoch / len(train_loader.dataset)
        accuracy = total_correct / total_samples

        print(f'Average Train Loss Per Batch: {train_loss_tl_avg} \n Train accuracy is: {accuracy}')
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
        # todo: change the data from dataloader based on the mimo_trainer
        with torch.no_grad():
            for batch_idx, validation_data in validation_loader_tqdm:
                data = validation_data[0]
                data = data.to(device)
                true_targets = (validation_data[3]).to(device)
                valid_mask = (true_targets != 0)
                valid_mask_expanded = valid_mask.unsqueeze(-1).expand(-1, -1, 3)
                masked_true_targets = true_targets[valid_mask]
                results, classifier_output = self.model(data)
                masked_classifier_output = classifier_output[valid_mask_expanded].view(-1, 3)
                classification_loss = loss_fn_classifier(masked_classifier_output, masked_true_targets)
                loss_vrnn = (kld_beta * results.kld_loss) + results.nll_loss
                loss = classification_loss

                total_loss += loss.item()
                reconstruction_loss_epoch += results.rec_loss.item()

                predicted_labels = torch.argmax(masked_classifier_output, dim=1)
                correct_predictions = (predicted_labels == masked_true_targets).sum().item()

                total_correct += correct_predictions
                total_samples += masked_true_targets.size(0)
                z_latent = torch.stack(results.z_latent, dim=2)
        validation_loss_avg = total_loss / len(validation_loader)
        reconstruction_loss_avg = reconstruction_loss_epoch / len(validation_loader.dataset)
        kld_loss_avg = kld_loss_epoch / len(validation_loader.dataset)
        accuracy = total_correct / total_samples

        print(f'Average Validation Loss Per Batch: {validation_loss_avg} \n Validation accuracy is: {accuracy}')
        return validation_loss_avg, accuracy

    #todo: figure out a unified way of saving the model checkpoint
    def do_train_with_dataset(self, train_loader, validation_loader):
        optimizer = torch.optim.Adam(list(self.model.vrnn_model.parameters()) +
                                     list(self.model.classifier_model.parameters()),
                                     lr=self.lr)
        # todo: Handle the best possible way to save checkpoints, because you are doing it in two places
        loss_fn = nn.CrossEntropyLoss()
        schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.lr_milestones)
        if self.previous_check_point is not None:
            print(f"Loading checkpoint '{self.previous_check_point}'")
            checkpoint = torch.load(self.previous_check_point)
            start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{self.previous_check_point}' (epoch {checkpoint['epoch']})")
        else:
            start_epoch = 1
        train_loss_list = []
        train_accuracy_list = []

        validation_loss_list = []
        validation_accuracy_list = []

        for epoch in tqdm(range(start_epoch, self.epochs_num + 1), desc='Epoch:'):
            train_loss, train_accuracy = self.train(epoch_train=epoch,
                                                    plot_dir=self.train_results_dir,
                                                    plot_every_epoch=self.plot_every_epoch,
                                                    train_loader=train_loader,
                                                    kld_beta=self.kld_beta_,
                                                    optimizer=optimizer,
                                                    loss_fn_classifier=loss_fn)
            schedular.step()
            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)
            validation_loss, validation_accuracy = self.validate_model(epoch_validation=epoch,
                                                                       validation_loader=validation_loader,
                                                                       loss_fn_classifier=loss_fn,
                                                                       kld_beta=1.1,
                                                                       plot_dir=None, tag='',
                                                                       plot_every_epoch=self.plot_every_epoch)
            validation_loss_list.append(validation_loss)
            validation_accuracy_list.append(validation_accuracy)

        history_dict = {
            'loss': train_loss_list,
            'accuracy': train_accuracy_list,
            'val_loss': validation_loss_list,
            'val_accuracy': validation_accuracy_list
        }
        return history_dict
        # todo You need to return the validation loss as well

    def predict_dataset(self, dataloader=None, model=None, model_name='default'):
        if self.model is None:
            self.model = model
        # todo: how to load the checkpoint at inference when you want?
        self.model.eval()
        all_predictions = []
        data_loader_tqdm = tqdm(enumerate(dataloader), total=len(dataloader))
        with torch.no_grad():
            for batch_idx, batched_data in data_loader_tqdm:
                data = batched_data[0].to(device)
                _, logits = self.model(data)
                predictions_ = torch.softmax(logits, dim=-1)
                all_predictions.append(predictions_.cpu().detach().numpy())
        all_predictions = np.concatenate(all_predictions, axis=0)
        return all_predictions


if __name__ == '__main__':
    config_file_path = r'config_arguments.yaml'
    graph_model = GraphModel(config_file_path=config_file_path)
    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    dataset_dir = os.path.normpath(config['dataset_config']['dataset_dir'])
    aux_dataset_hie_dir = os.path.normpath(config['dataset_config']['aux_dataset_dir'])
    stat_path = os.path.normpath(config['dataset_config']['stat_path'])
    batch_size = config['general_config']['batch_size']['train']
    fhr_healthy_dataset = JsonDatasetPreload(dataset_dir)
    dataset_size = len(fhr_healthy_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    print(f'Train size: {train_size} \n Test size: {test_size}')
    graph_model.setup_config()
    graph_model.create_model()
    train_dataset, test_dataset = random_split(fhr_healthy_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    print(f'Train size: {len(train_dataset)} \n Test size: {len(test_dataset)}')
    print('==' * 50)
    history_ = graph_model.do_train_with_dataset(train_loader, test_loader)
    predictions = graph_model.predict_dataset(dataloader=train_loader)
    print('done')
