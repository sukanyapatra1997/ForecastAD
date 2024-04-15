#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 15-Jan-2024
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the code for the forecast model.
# ---------------------------------------------------------------------------

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type
from .image_enc import Encoder, Decoder
from .time_enc import SinusoidalEncoding

class ForecastModel(nn.Module):

    def __init__(self, model_args: dict):
        """Initialization function

        Args:
            model_args (dict): model arguments.
        """
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        self.lstm_input_dim = model_args['lstm_input_dim']
        self.lstm_hidden_dim = model_args['lstm_hidden_dim']
        self.lstm_proj_dim = model_args['lstm_proj_dim']
        self.lstm_num_layers = model_args['lstm_num_layers']
        self.seq_len = model_args['seq_len']
        self.indim = model_args['indim']
        self.repdim = model_args['repdim']
        self.time_emb = model_args['time_emb']

        self.lstm = nn.LSTM(input_size=self.lstm_input_dim,
                            hidden_size=self.lstm_hidden_dim,
                            proj_size=self.lstm_proj_dim,
                            num_layers=self.lstm_num_layers,
                            batch_first=True)

        self.image_decoder = Decoder(input_size=self.indim, rep_dim=self.repdim)

        self.image_encoder = Encoder(input_size=self.indim, rep_dim=self.repdim)

        self.time_encoder = SinusoidalEncoding(emb_dim = self.time_emb)

    def forward(self,
                feature_data: Type[torch.tensor],
                feature_diff: Type[torch.tensor],
                feature_startdiff: Type[torch.tensor],
                target_diff: Type[torch.tensor],
                target_startdiff: Type[torch.tensor]
                ):
        """Forward pass of the forecast model

        Args:
            feature_data (Type[torch.tensor]): sequence of K feature tensors.
            feature_diff (Type[torch.tensor]): sequence of K interarrival times.
            feature_startdiff (Type[torch.tensor]): sequence of K duration from start.
            target_diff (Type[torch.tensor]): duration after which the prediction is to be made.
            target_startdiff (Type[torch.tensor]): duration of the target sample from the start of the operation.
            difffeature (str, optional): time embeddings to be used. Defaults to 'diff'.

        Returns:
            _type_: forecasted sample
        """

        # Image encoding
        batch_size, seq_length, c, h, w = feature_data.shape
        feature_data = feature_data.view(batch_size * seq_length, c, h, w)
        feature_data = self.image_encoder(feature_data)
        feature_data = feature_data.view(batch_size, seq_length, -1)

        # Time encoding: Feature
        batch_size, seq_length, d = feature_diff.shape
        feature_diff = feature_diff.view(batch_size * seq_length, d)
        feature_diff = self.time_encoder(feature_diff)
        feature_diff = feature_diff.view(batch_size, seq_length, -1)

        batch_size, seq_length, d = feature_startdiff.shape
        feature_startdiff = feature_startdiff.view(batch_size * seq_length, d)
        feature_startdiff = self.time_encoder(feature_startdiff)
        feature_startdiff = feature_startdiff.view(batch_size, seq_length, -1)

        diff = feature_diff + feature_startdiff

        history, _ = self.lstm(torch.cat([feature_data, diff], dim=-1))
        batch_size, seq_length, d = history.shape
        history = history[:,-1,:].contiguous()

        # Time encoding: Target
        target_diff = self.time_encoder(target_diff)

        target_startdiff = self.time_encoder(target_startdiff)

        diff2 = target_diff + target_startdiff

        out = torch.cat([history, diff2], dim=-1)
        out = self.image_decoder(out)

        return out
