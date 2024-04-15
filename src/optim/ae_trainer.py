#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 15-Jan-2024
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the code to train and test the AE model.
# ---------------------------------------------------------------------------

import logging
import time
import torch
import copy
import json
import numpy as np

from model import Autoencoder
from utils import Config
from pathlib import Path
from typing import Type
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve
)


class AETrainer():

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
        self.device = config.settings['device']

        self.model_name = 'aemodel_' + config.settings['experiment']
        self.model_path = config.settings['model_path']
        self.seq_len = config.settings['seq_len']
        self.output_path = config.settings['output_path']

        self.init_model()

    def init_model(self):
        """Initialize model
        """

        indim = self.config.settings['indim']
        repdim = self.config.settings['repdim']

        self.model = Autoencoder(input_size=indim, rep_dim=repdim).to(self.device)

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

        start_time = time.time()

        for epoch in range(self.n_epochs):

            train_loss_batch = []
            val_loss_batch = []

            for input in train_loader:

                optimizer.zero_grad()

                feature_data, _ = input
                feature_data = feature_data.to(self.device)

                pred_data = self.model(feature_data)

                scores = torch.sum((pred_data - feature_data) ** 2, dim=tuple(range(1, feature_data.dim())))

                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                train_loss_batch.append(loss.item())

            avg_train_loss = sum(train_loss_batch) / len(train_loss_batch)

            with torch.no_grad():
                for input in val_loader:

                    feature_data, _, _ = input
                    feature_data = feature_data.to(self.device)

                    pred_data = self.model(feature_data)

                    scores = torch.sum((pred_data - feature_data) ** 2, dim=tuple(range(1, feature_data.dim())))
                    loss = torch.mean(scores)

                    val_loss_batch.append(loss.item())

            avg_val_loss = sum(val_loss_batch) / len(val_loss_batch)

            self.logger.info("Epoch %d: train MSE %.4f, val MSE %.4f" % (epoch, avg_train_loss, avg_val_loss))

            # Implement early stopping
            if self.early_stopping:

                if epoch == 0 :
                    min_loss = avg_val_loss
                    best_model = copy.deepcopy(self.model)
                    patience_cnt = 0
                    continue

                if avg_val_loss < min_loss:
                    min_loss = avg_val_loss
                    patience_cnt = 0

                    best_model = copy.deepcopy(self.model)
                else:
                    patience_cnt +=1
                    if patience_cnt == self.patience:
                        self.logger.info('Training stops at {} epoch'.format(epoch+1))
                        break

        train_time = time.time() - start_time

        self.logger.info('Finished training AE. Total time: %.3f' % train_time)

        if self.early_stopping:
            self.model = copy.deepcopy(best_model)

        model_path = self.save_model(self.model_name)

        return model_path

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
                feature_data, timestamps, labels = input
                feature_data = feature_data.to(self.device)

                pred_data = self.model(feature_data)

                scores = torch.sum((pred_data - feature_data) ** 2, dim=tuple(range(1, feature_data.dim())))
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

        export_path_met = Path(self.config.settings['output_path']).joinpath('Ae_test_metrics.json')
        with open(export_path_met, 'w') as fp:
            json.dump({
                "auc": auc(fpr, tpr) * 100,
                "aupr": auc(recall, precision) * 100
            }, fp)

        test_time = time.time() - start_time

        self.logger.info('AE testing time: %.3f' % test_time)
        self.logger.info('Finished testing AE.')


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
        self.logger.info(f"Saving model to file {export_path.as_posix()}")

        return export_path