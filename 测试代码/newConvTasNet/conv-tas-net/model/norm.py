from functools import partial
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from typing import List


EPS = 1e-8


def z_norm(x, dims: List[int], eps: float = 1e-8):
    mean = x.mean(dim=dims, keepdim=True)
    var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
    value = (x - mean) / torch.sqrt((var2 + eps))
    return value


def _glob_norm(x, eps: float = 1e-8):
    dims: List[int] = torch.arange(1, len(x.shape)).tolist()
    return z_norm(x, dims, eps)




class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x, EPS: float = 1e-8):
        """Applies forward pass.
        Works for any input size > 2D.
        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`
        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        value = _glob_norm(x, eps=EPS)
        return self.apply_gain_and_bias(value)
