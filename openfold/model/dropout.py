import torch
import torch.nn as nn
import json
import os
from functools import partialmethod
from typing import Union, List, Optional


# Doing this in a jank way to prevent conflicts with OpenFold's checkpointing
config_path = os.path.join('..', '..', 'optimize', 'tmp_dropout_config.json')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

NOISE_FACTOR = config.get('NOISE_FACTOR', 0.1)  
BINARIZE = config.get('BINARIZE', False)
DROPOUT_RATE = config.get('DROPOUT_RATE', 0.5)



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
        # print(f"any_nan pre: {torch.any(torch.isnan(x))}")
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        if mask is None:
            shape = list(x.shape)
            if self.batch_dim is not None:
                for bd in self.batch_dim:
                    shape[bd] = 1
            mask = x.new_ones(shape)
            mask = self.dropout(mask)
        else:
            correct_view_mask = mask.view(shape)

            noise = torch.randn_like(correct_view_mask) * correct_view_mask.std() * NOISE_FACTOR
            noised_mask = correct_view_mask + noise

            sigmoided_mask = torch.sigmoid(noised_mask)

            if(BINARIZE):
                sorted_values, indices = torch.sort(sigmoided_mask.view(-1), descending=False)
                threshold = sorted_values[int(sorted_values.numel() * DROPOUT_RATE)]

                hard_mask = (sigmoided_mask > threshold).float()
                extremified_mask = sigmoided_mask + (hard_mask - sigmoided_mask).detach()
            else:
                extremified_mask = sigmoided_mask


            dropout_rate = 1 - torch.mean(extremified_mask)


            dropped_x = x * extremified_mask
            dropped_x *= 1 / (1 - dropout_rate)
            x = dropped_x

            return x

        x *= mask

        
        return x


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