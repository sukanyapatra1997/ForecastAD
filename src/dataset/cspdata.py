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

class CSP_Dataset():
    """Implementation of CSP Dataset
    """

    def __init__(self, config: Config):
        """Class constructor

        Args:
            config (Config): configuration file
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        transform = transforms.ToTensor()

        self.train_set = CSP(config = self.config, transform=transform)
        self.val_set = CSP(config = self.config, transform=transform, mode = 'val')
        self.test_set = CSP(config = self.config, transform=transform, mode = 'test')

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

class CSP(Dataset):
    """Base class for CSP dataset
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

        self.data_seq, self.diff_seq, self.startdiff_seq, self.label_seq = self.__make_dataset__()

    def __make_dataset__(self):

        file_list = {
            'image_path': [],
            'timestamps': [],
            'days': []
        }


        p = self.datapath.glob('*.*')

        imgpaths = [x for x in p if x.is_file()]

        file_list['image_path'].extend(imgpaths)
        file_list['timestamps'].extend([int(x.stem) for x in imgpaths])
        file_list['days'].extend([datetime.date.fromtimestamp(int(x.stem)/1000) for x in imgpaths])

        # Sort samples based on timestamp
        timestamps = []
        imgpaths = []
        days = []

        for i,x,j in tqdm(sorted(zip(file_list['timestamps'], file_list['days'], file_list['image_path']))):
            timestamps.append(i)
            imgpaths.append(j)
            days.append(x)

        file_list = {
            'image_path': imgpaths,
            'timestamps': timestamps,
            'days': np.array(days)
        }

        # Free Space
        timestamps = []
        imgpaths = []
        days = []

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

        # initialize arrays
        data_seq, diff_seq, startdiff_seq, label_seq  = [], [], [], []

        # loop over samples
        for sample_idx, sample in enumerate(tqdm(samples, disable=DISABLE_TQDM)):

            # get the index for the current sample
            sample_ind = np.where(np.array(file_list['timestamps']) == int(sample))[0][0]

            # get seq_len + 2 samples where 1 (previous) + seq_len (context) + 1 (target)
            currtimeseq = file_list['timestamps'][sample_ind - (self.seq_len + 1) : sample_ind + 1]

            currtimediff, currentstartdiff = [], []

            # get the date of the current sample
            seq_date = datetime.date.fromtimestamp(currtimeseq[-1]/1000)

            # compute the interarrival times
            VALIDSEQ = True
            validfrom = 0
            INCORRPREV = False

            for idx, ts in enumerate(currtimeseq):

                if idx == 0:
                    prevtimestamp = ts
                    continue

                if (datetime.date.fromtimestamp(ts/1000) == seq_date):

                    if INCORRPREV:
                        currtimediff.append(EPS)
                        INCORRPREV = False
                        validfrom = idx - 1

                    else:
                        currtimediff.append((datetime.datetime.fromtimestamp(ts/1000) -
                                datetime.datetime.fromtimestamp(prevtimestamp/1000)).total_seconds())

                else:
                    currtimediff.append(EPS)
                    INCORRPREV = True
                    VALIDSEQ = False

                prevtimestamp = ts

            # list of file paths
            currdateseq = file_list['image_path'][sample_ind - self.seq_len : sample_ind + 1]

            # either recify by repeating or discard invalid sequences
            if not VALIDSEQ:

                if RECTIFYSEQ:
                    for idx in range(len(currdateseq)):
                        if idx < validfrom:
                            currdateseq[idx] = currdateseq[validfrom]
                            currtimeseq[idx + 1] = currtimeseq[validfrom + 1]
                else:
                    continue

            data_seq.append(currdateseq)
            diff_seq.append(currtimediff)

            # get interval between the current timestamp and the begining of the day
            starttime = np.where(file_list['days'] == seq_date)[0][0]
            starttime = datetime.datetime.fromtimestamp(file_list['timestamps'][starttime]/1000)
            startdiff_seq.append([max(EPS, (datetime.datetime.fromtimestamp(ts/1000) - starttime).total_seconds()) for ts in currtimeseq[1:]])

            if self.mode != 'train':
                label_seq.append(labels[sample_idx])

        return data_seq, diff_seq, startdiff_seq, label_seq

    def __len__(self):

        return len(self.data_seq)

    def __getitem__(self, idx):

        Tensor = torch.LongTensor

        if torch.is_tensor(idx):
            idx = idx.tolist()

        files = self.data_seq[idx]
        feature_diff = torch.tensor(self.diff_seq[idx][:-1]).reshape(-1, 1)
        feature_startdiff = torch.tensor(self.startdiff_seq[idx][:-1]).reshape(-1, 1)

        target_diff = torch.tensor([self.diff_seq[idx][-1]])
        target_startdiff = torch.tensor([self.startdiff_seq[idx][-1]])

        feature_data = []
        target_data = None

        for i, file in enumerate(files):

            with open(file, 'rb') as fp:
                image = pickle.load(fp)

            image = image.astype(np.float32)
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            image = cv2.resize(image, dsize=(self.indim, self.indim), interpolation=cv2.INTER_AREA)

            if self.transform:
                image = self.transform(image).reshape(3,self.indim,self.indim)

            if i == len(files) - 1:
                target_data = image
            else:
                feature_data.append(image.unsqueeze(0))

        if self.mode == "train":
            return torch.cat(feature_data, dim=0), feature_diff, feature_startdiff,\
                target_data, target_diff, target_startdiff, Tensor([int(files[-1].stem)])

        else:

            return torch.cat(feature_data, dim=0), feature_diff, feature_startdiff,\
                target_data, target_diff, target_startdiff, Tensor([int(files[-1].stem)]), Tensor([int(self.label_seq[idx])])
