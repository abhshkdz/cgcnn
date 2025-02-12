"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py
"""

from __future__ import annotations

import torch
from torch import nn


def stochastic_depth_2d(
    input: torch.Tensor, batch: torch.Tensor, p: float, training: bool = True
) -> torch.Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[num_nodes, ...]): The input tensor or arbitrary dimensions
            with the first one being its node dimension.
        batch (LongTensor[num_nodes]): The batch tensor of the input tensor.
        p (float): probability of the input to be zeroed.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if not training or p == 0.0:
        return input

    batch_size = batch.max() + 1
    survival_rate = 1.0 - p
    size = [batch_size] + [1] * (input.ndim - 1)
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise[batch]


# Exact same copy to make torch compile happy, or it will recompile...
def stochastic_depth_3d(
    input: torch.Tensor, batch: torch.Tensor, p: float, training: bool = True
) -> torch.Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if not training or p == 0.0:
        return input

    batch_size = batch.max() + 1
    survival_rate = 1.0 - p
    size = [batch_size] + [1] * (input.ndim - 1)
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise[batch]


class StochasticDepth(nn.Module):
    """
    Stochastic Depth for graph features.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, node_features, edge_features, node_batch):
        # Ensure contiguous layout to make torch compile happy
        node_features = node_features.contiguous()
        edge_features = edge_features.contiguous()

        node_features = stochastic_depth_2d(
            input=node_features, batch=node_batch, p=self.p, training=self.training
        )
        edge_features = stochastic_depth_3d(
            input=edge_features, batch=node_batch, p=self.p, training=self.training
        )
        return node_features, edge_features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class SkipStochasticDepth(nn.Module):
    """
    Skip Stochastic Depth for graph features.
    """

    def forward(self, node_features, edge_features, _):
        return node_features, edge_features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p=0.0)"
