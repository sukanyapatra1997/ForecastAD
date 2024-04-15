#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 15-Jan-2024
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the code for time encoding
# ---------------------------------------------------------------------------

# https://github.com/babylonhealth/neuralTPPs/blob/master/tpp/utils/encoding.py

import torch
import torch.nn as nn
import numpy as np

class SinusoidalEncoding(nn.Module):
    def __init__(self, emb_dim, scaling: float = 1.):
        super(SinusoidalEncoding, self).__init__()
        self.emb_dim = emb_dim
        self.scaling = scaling

    def forward(
            self,
            times,
            min_timescale: float = 1.0,
            max_timescale: float = 1000
    ):
        """
        Adaptation of positional encoding to include temporal information
        """

        assert self.emb_dim % 2 == 0, "hidden size must be a multiple of 2 " \
                                      "with pos_enc, pos_dec"
        num_timescales = self.emb_dim // 2
        log_timescale_increment = np.log(max_timescale / min_timescale
                                         ) / (num_timescales - 1)
        inv_timescales = (
                min_timescale * torch.exp(
                    torch.arange(
                        num_timescales, dtype=torch.float, device=times.device
                    ) * -log_timescale_increment))
        scaled_time = times.type(
            torch.FloatTensor).to(times.device) * inv_timescales.unsqueeze(
            0) * self.scaling
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        return signal
