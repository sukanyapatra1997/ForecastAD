#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 15-Jan-2024
# Updated Date: 20-Jun-2024
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the code to train and test the forecast model.
# ---------------------------------------------------------------------------

import logging
import time
import torch
import copy
import json
import numpy as np

from model import ForecastModel, Encoder, Decoder, Autoencoder
from utils import Config
from pathlib import Path
from typing import Type
from sklearn.metrics import (
    auc,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)


class ForecastTrainer():

    def __init__(self, config: Type[Config]):
        """Init function

        Args:
            config (Type[Config]): experimental configuration file.
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config
        self.optimizer_name = config.settings['optimizer_name']
        self.early_stopping = config.settings['early_stopping']
        self.lr = config.settings['learning_rate']
        self.n_epochs = config.settings['epochs']
        self.patience = config.settings['patience']
        self.lr_milestones = config.settings['learning_rate_milestones']
        self.batch_size = config.settings['batch_size']
        self.weight_decay = config.settings['weight_decay']
        self.n_jobs_dataloader = config.settings['num_workers']
        self.difffeature = config.settings['difffeature']
        self.device = config.settings['device']

        self.model_name = 'forecastmodel_' + config.settings['experiment']
        self.model_path = config.settings['model_path']
        self.seq_len = config.settings['seq_len']
        self.output_path = config.settings['output_path']

        self.init_model()

    def init_model(self):
        """Initialize model
        """

        # Default params
        lstm_input_dim = 48
        lstm_hidden_dim = 32
        lstm_proj_dim = 16

        indim = self.config.settings['indim']
        repdim = self.config.settings['repdim']
        time_emb = repdim // 2

        indiff, outdiff = self.difffeature.split("_")
        assert indiff in ['no', 'tau', 'delta', 'all']

        if indiff == 'no':
            lstm_input_dim = repdim
        else:
            lstm_input_dim = repdim + time_emb

        assert outdiff in ['no', 'tau', 'delta', 'all']
        if outdiff == 'no':
            lstm_proj_dim = repdim
            lstm_hidden_dim = repdim + time_emb
        else:
            lstm_hidden_dim = repdim
            lstm_proj_dim = repdim // 2

        model_args = {
            'lstm_input_dim': lstm_input_dim,
            'lstm_hidden_dim': lstm_hidden_dim,
            'lstm_proj_dim': lstm_proj_dim,
            'lstm_num_layers': self.config.settings['llayers'] if 'llayers' in self.config.settings else 2,
            'seq_len': self.seq_len,
            'indim': indim,
            'repdim': repdim,
            'time_emb_in': time_emb,
            'time_emb_out': time_emb,
        }

        self.logger.info(model_args)

        self.model = ForecastModel(model_args).to(self.device)

        # If running for testing always load the trained model
        if not self.config.settings['train_forecast']:
            self.config.settings['load_forecast'] = True

        if self.config.settings['load_forecast']:
            self.logger.info('Loading forecast model')
            self.load_model(self.model_name)
            self.config.settings['pretrained_ae'] = None

        if self.config.settings['pretrained_ae']:
            self.logger.info('Loading pretrained AE for forecast model')
            self.model.image_encoder, self.model.image_decoder = self.init_network_weights_from_pretrainin(
                                                                    self.model.image_encoder,
                                                                    self.model.image_decoder,
                                                                    self.config.settings['pretrained_ae'],
                                                                    input_size=indim,
                                                                    rep_dim=repdim
                                                                )

    def train(self, dataset):
        """Method for model training.

        Args:
            dataset (CSP_Dataset): dataset to be used for training.
        """

        self.logger.info('Starting training...')
        train_loader, val_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        self.model.to(device=self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr = self.lr,
                                     weight_decay = self.weight_decay)


        for epoch in range(self.n_epochs):

            train_loss_batch = []
            val_loss_batch = []

            for input in train_loader:

                optimizer.zero_grad()


                feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, _ = input

                feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff = feature_data.to(self.device),\
                                                                                                            feature_diff.to(self.device),\
                                                                                                            feature_startdiff.to(self.device),\
                                                                                                            target_data.to(self.device),\
                                                                                                            target_diff.to(self.device),\
                                                                                                            target_startdiff.to(self.device)


                pred_data = self.model(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature=self.difffeature)

                score = torch.sum((pred_data - target_data) ** 2, dim=tuple(range(1, target_data.dim())))
                loss = torch.mean(score)
                loss.backward()
                optimizer.step()

                train_loss_batch.append(loss.item())

            avg_train_loss = sum(train_loss_batch) / len(train_loss_batch)

            val_scores = []
            val_labels = []

            with torch.no_grad():
                for input in val_loader:
                    feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, _, labels = input

                    feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff = feature_data.to(self.device),\
                                                                                                                feature_diff.to(self.device),\
                                                                                                                feature_startdiff.to(self.device),\
                                                                                                                target_data.to(self.device),\
                                                                                                                target_diff.to(self.device),\
                                                                                                                target_startdiff.to(self.device)

                    pred_data = self.model(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature=self.difffeature)

                    score = torch.sum((pred_data - target_data) ** 2, dim=tuple(range(1, target_data.dim())))

                    val_scores.extend(score.cpu().data.numpy().tolist())
                    val_labels.extend(labels.cpu().data.numpy().tolist())

                    loss = torch.mean(score)

                    val_loss_batch.append(loss.item())

            avg_val_loss = sum(val_loss_batch) / len(val_loss_batch)
            val_auc = roc_auc_score(np.array(val_labels), np.array(val_scores))


            self.logger.info("Epoch %d: train MSE %.4f, val MSE %.4f, val AUC %.4f" % (epoch, avg_train_loss, avg_val_loss, val_auc))

            # Implement early stopping
            if self.early_stopping:

                if epoch == 0 :
                    max_auc = val_auc
                    best_model = copy.deepcopy(self.model)
                    patience_cnt = 0
                    continue

                if val_auc > max_auc:
                    max_auc = val_auc
                    patience_cnt = 0

                    best_model = copy.deepcopy(self.model)
                else:
                    patience_cnt +=1
                    if patience_cnt == self.patience:
                        self.logger.info('Training stops at {} epoch'.format(epoch+1))
                        break

        self.logger.info('Finished training.')

        if self.early_stopping:
            self.model = copy.deepcopy(best_model)

        self.save_model(self.model_name)

    def test(self, dataset):
        """Method for model evaluation.

        Args:
            dataset (CSP_Dataset): dataset to be used for test.
        """

        self.logger.info('Start testing.')
        self.model.eval()

        if not Path.exists(Path(self.config.settings['output_path'])):
            Path.mkdir(Path(self.config.settings['output_path']), parents=True)

        # Get test data loader
        _, _, test_loader = dataset.loaders(batch_size = self.config.settings['batch_size'],
                                            num_workers = self.config.settings['num_workers'])

        loss_epoch = 0.0

        start_time = time.time()

        # Calculate anomaly score for test samples
        test_scores = []
        true_labels = []

        with torch.no_grad():
            for input in test_loader:
                feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff, timestamps, labels = input

                feature_data, feature_diff, feature_startdiff, target_data, target_diff, target_startdiff = feature_data.to(self.device),\
                                                                                                            feature_diff.to(self.device),\
                                                                                                            feature_startdiff.to(self.device),\
                                                                                                            target_data.to(self.device),\
                                                                                                            target_diff.to(self.device),\
                                                                                                            target_startdiff.to(self.device)

                pred_data = self.model(feature_data, feature_diff, feature_startdiff, target_diff, target_startdiff, difffeature=self.difffeature)

                scores = torch.sum((pred_data - target_data) ** 2, dim=tuple(range(1, target_data.dim())))
                loss = torch.mean(scores)

                test_scores.extend(scores.cpu().data.numpy().tolist())
                true_labels.extend(labels.cpu().data.numpy().tolist())
                loss_epoch += loss.item()


        self.logger.info('Test set Loss: {:.8f}'.format(loss_epoch / len(test_loader)))

        # Sort samples based on anomaly score
        scores = []
        labels = []

        for scr, tl in sorted(zip(test_scores, true_labels)):
            scores.append(scr)
            labels.append(tl)


        # Save test metrics
        fpr, tpr, _ = roc_curve(np.array(labels), np.array(scores))
        precision, recall, _ = precision_recall_curve(np.array(labels), np.array(scores))


        export_path_met = Path(self.config.settings['output_path']).joinpath('test_metrics.json')
        with open(export_path_met, 'w') as fp:
            json.dump({
                "auc": auc(fpr, tpr) * 100,
                "aupr": auc(recall, precision) * 100
            }, fp)

        test_time = time.time() - start_time

        self.logger.info('ForecastAD testing time: %.3f' % test_time)
        self.logger.info('Finished testing ForecastAD.')



    def save_model(self, name: Type[str]):
        """Method to save the trained models
        Args:
            name (str): name of the saved model file
        """

        if not Path.exists(Path(self.model_path)):
            Path.mkdir(Path(self.model_path), parents=True, exist_ok=True)

        model_dict = self.model.state_dict()

        export_path = Path(self.model_path).joinpath(name + '.tar')

        torch.save(model_dict, export_path)
        self.logger.info("Saving model to file")

    def load_model(self, name: Type[str]):
        """Method to load saved model
        Args:
            name (str): name of the saved model file
        """
        self.logger.info("Loading model from file")
        import_path = Path(self.model_path).joinpath(name + '.tar')

        if not Path.exists(import_path):
            raise ValueError('Model checkpoint path is invalid')

        load_dict = torch.load(import_path, map_location=torch.device('cpu'))

        forcast_dict = self.model.state_dict()
        updated_dict = {k: v for k, v in load_dict.items() if k in forcast_dict}

        self.model.load_state_dict(updated_dict)

    def init_network_weights_from_pretrainin(self,
                                             image_encoder: Type[Encoder],
                                             image_decoder: Type[Decoder],
                                             pretrain_path: Type[str],
                                             input_size: int,
                                             rep_dim: int):

        self.logger.info(f"Initialising Image encoder {pretrain_path}")

        pretrained_model = Autoencoder(input_size=input_size, rep_dim=rep_dim)
        if not Path.exists(Path(pretrain_path)):
            raise ValueError('Model checkpoint path is invalid')

        pretrained_model.load_state_dict(torch.load(Path(pretrain_path)))
        pretrained_model_dict = pretrained_model.state_dict()


        image_encoder_dict = image_encoder.state_dict()
        updated_dict = {k: v for k, v in pretrained_model_dict.items() if k in image_encoder_dict}
        image_encoder_dict.update(updated_dict)
        image_encoder.load_state_dict(image_encoder_dict)

        decoder_dict = image_decoder.state_dict()
        updated_dict = {k: v for k, v in pretrained_model_dict.items() if k in decoder_dict}
        decoder_dict.update(updated_dict)
        image_decoder.load_state_dict(decoder_dict)

        return image_encoder, image_decoder

