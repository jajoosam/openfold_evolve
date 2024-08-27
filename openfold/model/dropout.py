import torch
import torch.nn as nn
import json
import os
from functools import partialmethod
from typing import Union, List, Optional
import logging
import numpy as np

# Doing this in a jank way to prevent conflicts with OpenFold's checkpointing
config_path = "tmp_dropout_config.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

NOISE_FACTOR = config.get('NOISE_FACTOR', 0.1)  
BINARIZE = config.get('BINARIZE', False)
DROPOUT_RATE = config.get('DROPOUT_RATE', 0.15)
DISABLED = config.get('DISABLED', False)



def noise(shape, noise_factor):
    noise = np.random.randn(*shape) * noise_factor
    noise_torch = torch.from_numpy(noise).float()
    return noise_torch

class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        config_path = "tmp_dropout_config.json"
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

            self.noise_factor = config.get('NOISE_FACTOR', 0.1)  
            self.binarize = config.get('BINARIZE', False)
            self.dropout_rate = config.get('DROPOUT_RATE', 0.05)
            self.disabled = config.get('DISABLED', False)

        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
            mask:
                Optional pre-determined dropout mask. If provided, it should be
                of the same shape as `x` or broadcastable to `x`.
        """
        if(self.disabled or mask is None):
            return x
        # else:
        #     print(f"Got mask {mask.shape}")
        shape = list(x.shape)

        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1

        correct_view_mask = mask.view(shape)
        noised_mask = correct_view_mask + noise(correct_view_mask.shape, self.noise_factor).to(correct_view_mask.device)

        if(self.binarize):
            sorted_values, indices = torch.sort(noised_mask.view(-1), descending=False)
            threshold = sorted_values[int(sorted_values.numel() * self.dropout_rate)]

            hard_mask = (noised_mask > threshold).float()
            extremified_mask = noised_mask + (hard_mask - noised_mask).detach()

        else:
            extremified_mask = noised_mask
        


        dropout_rate = 1 - torch.mean(extremified_mask)

        dropped_x = x * extremified_mask

        dropped_x *= 1 / (1 - dropout_rate)

        return dropped_x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)