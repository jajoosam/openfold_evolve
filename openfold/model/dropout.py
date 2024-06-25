import torch
import torch.nn as nn
from functools import partialmethod
from typing import Union, List, Optional

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
            # binarize the mask
            mask = torch.sigmoid(mask)
            # since we're using the mask, we must also scale the input by the dropout rate
            dropout_rate = 1 - torch.mean(mask)
            dropped_x = x * mask
            dropped_x *= 1 / (1 - dropout_rate)
            # print(f"Scaling by {1 / (1 - dropout_rate)}")
            # print(f"any_nan post: {torch.any(torch.isnan(x))}")
            return dropped_x

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