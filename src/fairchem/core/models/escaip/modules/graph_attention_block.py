from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from fairchem.core.models.escaip.configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        MolecularGraphConfigs,
        RegularizationConfigs,
    )
    from fairchem.core.models.escaip.custom_types import GraphAttentionData

from fairchem.core.models.escaip.modules.base_block import BaseGraphNeuralNetworkLayer
from fairchem.core.models.escaip.utils.nn_utils import (
    NormalizationType,
    get_feedforward,
    get_linear,
    get_normalization_layer,
)
from fairchem.core.models.escaip.utils.stochastic_depth import (
    SkipStochasticDepth,
    StochasticDepth,
)


class EfficientGraphAttentionBlock(nn.Module):
    """
    Efficient Graph Attention Block module.
    Ref: swin transformer
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        # Graph attention
        self.graph_attention = EfficientGraphAttention(
            global_cfg=global_cfg,
            molecular_graph_cfg=molecular_graph_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
        )

        # Feed forward network
        self.feedforward = FeedForwardNetwork(
            global_cfg=global_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
        )

        # Normalization
        normalization = NormalizationType(reg_cfg.normalization)
        self.norm_attn = get_normalization_layer(normalization, is_graph=False)(
            global_cfg.hidden_size
        )
        self.norm_ffn = get_normalization_layer(normalization)(
            global_cfg.hidden_size, skip_edge=not global_cfg.direct_forces
        )

        # Stochastic depth
        self.stochastic_depth_attn = (
            StochasticDepth(reg_cfg.stochastic_depth_prob)
            if reg_cfg.stochastic_depth_prob > 0.0
            else SkipStochasticDepth()
        )
        self.stochastic_depth_ffn = (
            StochasticDepth(reg_cfg.stochastic_depth_prob)
            if reg_cfg.stochastic_depth_prob > 0.0
            else SkipStochasticDepth()
        )

    def forward(
        self,
        data: GraphAttentionData,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        # ref: swin transformer https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py#L452
        # x = x + self.stochastic_depth(self.graph_attention(self.norm_attn(x)))
        # x = x + self.stochastic_depth(self.feedforward(self.norm_ffn(x)))

        # attention
        node_hidden = self.norm_attn(node_features)
        node_hidden, edge_hidden = self.graph_attention(data, node_hidden)
        node_hidden, edge_hidden = self.stochastic_depth_attn(
            node_hidden, edge_hidden, data.node_batch
        )
        node_features, edge_features = (
            node_hidden + node_features,
            edge_hidden + edge_features,
        )

        # feedforward
        node_hidden, edge_hidden = self.norm_ffn(node_features, edge_features)
        node_hidden, edge_hidden = self.feedforward(node_hidden, edge_hidden)
        node_hidden, edge_hidden = self.stochastic_depth_ffn(
            node_hidden, edge_hidden, data.node_batch
        )
        node_features, edge_features = (
            node_hidden + node_features,
            edge_hidden + edge_features,
        )
        return node_features, edge_features


class EfficientGraphAttention(BaseGraphNeuralNetworkLayer):
    """
    Efficient Graph Attention module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__(global_cfg, molecular_graph_cfg, gnn_cfg, reg_cfg)

        # Edge linear layer
        self.edge_linear = self.get_edge_linear(gnn_cfg, global_cfg, reg_cfg)

        # Node hidden layer
        self.node_linear = self.get_node_linear(global_cfg, reg_cfg)

        # message linear
        self.message_linear = get_linear(
            in_features=global_cfg.hidden_size * 2,
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
        )

        # Multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=global_cfg.hidden_size,
            num_heads=gnn_cfg.atten_num_heads,
            dropout=reg_cfg.atten_dropout,
            bias=True,
            batch_first=True,
        )

        # scalar for attention bias
        self.use_angle_embedding = gnn_cfg.use_angle_embedding
        if self.use_angle_embedding:
            self.attn_scalar = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            self.attn_scalar = torch.tensor(1.0)

    def forward(
        self,
        data: GraphAttentionData,
        node_features: torch.Tensor,
    ):
        # Get edge features
        edge_features = self.get_edge_features(data)
        edge_hidden = self.edge_linear(edge_features)

        # Get node features
        node_features = self.get_node_features(node_features, data.neighbor_list)
        node_hidden = self.node_linear(node_features)

        # Concatenate edge and node features (num_nodes, num_neighbors, hidden_size)
        message = self.message_linear(torch.cat([edge_hidden, node_hidden], dim=-1))

        # Multi-head self-attention
        if self.use_angle_embedding:
            attn_mask = data.attn_mask + data.angle_embedding * self.attn_scalar
        else:
            attn_mask = data.attn_mask
        edge_output = self.multi_head_attention(
            query=message,
            key=message,
            value=message,
            # key_padding_mask=~data.neighbor_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]

        # Aggregation
        node_output = self.aggregate(edge_output, data.neighbor_mask)

        return node_output, edge_output


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()
        self.mlp_node = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.ffn_hidden_layer_multiplier,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

        if global_cfg.direct_forces:
            self.mlp_edge = get_feedforward(
                hidden_dim=global_cfg.hidden_size,
                activation=global_cfg.activation,
                hidden_layer_multiplier=gnn_cfg.ffn_hidden_layer_multiplier,
                bias=True,
                dropout=reg_cfg.mlp_dropout,
            )
        else:
            self.mlp_edge = nn.Identity()

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor):
        return self.mlp_node(node_features), self.mlp_edge(edge_features)
