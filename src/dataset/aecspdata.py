#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 15-Jan-2024
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the base class for the CSP dataset
# ---------------------------------------------------------------------------

import cv2
import time
import torch
import numpy as np
import logging
import json
import pickle
import datetime
import torchvision.transforms as transforms

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from utils import Config

DISABLE_TQDM = False
RECTIFYSEQ = True
EPS = 1e-5

class AECSP_Dataset():
    """Implementation of AECSP Dataset
    """

    def __init__(self, config: Config):
        """Class constructor

        Args:
            config (Config): configuration file
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        transform = transforms.ToTensor()

        self.train_set = AECSP(config = self.config, transform=transform)
        self.val_set = AECSP(config = self.config, transform=transform, mode = 'val')
        self.test_set = AECSP(config = self.config, transform=transform, mode = 'test')

        self.logger.info('Train samples: {} Val samples: {} Test samples: {}'
                        .format(len(self.train_set), len(self.val_set), len(self.test_set)))

        self.logger.info('Dataset Configured')

    def loaders(self, batch_size: int, shuffle=True, num_workers: int = 0):
        """Initialise data loaders

        Args:
            batch_size (int): batch size
            shuffle (bool, optional): shuffle the data samples. Defaults to True.
            num_workers (int, optional): number of concurrent workers. Defaults to 0.

        Returns:
            train_dataLoader (DataLoader): train data loader
            val_dataLoader (DataLoader): validation data loader
            test_dataLoader (DataLoader): test data loader
        """

        train_dataLoader = DataLoader(self.train_set, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers)
        val_dataLoader = DataLoader(self.val_set, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers)
        test_dataLoader = DataLoader(self.test_set, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers)
        return train_dataLoader, val_dataLoader, test_dataLoader

class AECSP(Dataset):
    """Base class for AECSP dataset
    """

    def __init__(self, config: Config, transform: transforms = None, mode: str = 'train'):
        """_summary_

        Args:
            root (Path): _description_
            data (dict): _description_
            device (str): _description_
            transform (transforms, optional): _description_. Defaults to None.
            seq_len (int, optional): _description_. Defaults to 4.
        """
        self.config = config
        self.indim = self.config.settings['indim']
        self.root = self.config.settings['data_path']
        self.experiment = self.config.settings['experiment']
        self.device = self.config.settings['device']
        self.transform = transform
        self.mode = mode
        self.seq_len = self.config.settings['seq_len']
        self.datapath = Path(self.root)

        self.data_seq, self.label_seq = self.__make_dataset__()

    def __make_dataset__(self):

        # load from JSON file
        if self.mode == 'train':
            with open(self.config.settings["train_set_path"], 'r') as fp:
                samples = json.load(fp)
            samples = samples['train']
            labels = None

        elif self.mode == 'val':
            with open(self.config.settings["val_set_path"], 'r') as fp:
                val_labels = json.load(fp)

            samples = val_labels['normal'].copy()
            samples.extend(val_labels['anomalous'].copy())
            labels = [0]*len(val_labels['normal'])
            labels.extend([1]*len(val_labels['anomalous']))

        else:
            with open(self.config.settings["test_set_path"], 'r') as fp:
                test_labels = json.load(fp)

            samples = test_labels['normal'].copy()
            samples.extend(test_labels['anomalous'].copy())
            labels = [0]*len(test_labels['normal'])
            labels.extend([1]*len(test_labels['anomalous']))

        return samples, labels

    def __len__(self):

        return len(self.data_seq)

    def __getitem__(self, idx):

        Tensor = torch.LongTensor

        if torch.is_tensor(idx):
            idx = idx.tolist()


        file = self.datapath.joinpath(str(self.data_seq[idx]) + '.pickle')

        with open(file, 'rb') as fp:
            image = pickle.load(fp)

        image = image.astype(np.float32)

        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = cv2.resize(image, dsize=(self.indim, self.indim), interpolation=cv2.INTER_AREA)

        if self.transform:
            image = self.transform(image).reshape(3,self.indim,self.indim)

        if self.mode == "train":
            return image, Tensor([int(self.data_seq[idx])])

        else:
            return image, Tensor([int(self.data_seq[idx])]), Tensor([self.label_seq[idx]])
