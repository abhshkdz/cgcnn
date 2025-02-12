from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

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
    get_feedforward,
    get_normalization_layer,
)


class InputBlock(nn.Module):
    """
    Wrapper of InputLayer for adding normalization
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        self.input_layer = InputLayer(global_cfg, molecular_graph_cfg, gnn_cfg, reg_cfg)

        self.norm = get_normalization_layer(reg_cfg.normalization)(
            global_cfg.hidden_size, skip_edge=not global_cfg.direct_forces
        )

    def forward(self, inputs: GraphAttentionData):
        node_features, edge_features = self.input_layer(inputs)
        return self.norm(node_features, edge_features)


class InputLayer(BaseGraphNeuralNetworkLayer):
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

        # ffn for edge features
        self.edge_ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=1,
            dropout=reg_cfg.mlp_dropout,
            bias=True,
        )

        # normalization
        self.norm = get_normalization_layer(reg_cfg.normalization, is_graph=False)(
            global_cfg.hidden_size
        )

    def forward(self, inputs: GraphAttentionData):
        # Get edge features
        edge_features = self.get_edge_features(inputs)

        # Edge processing
        edge_hidden = self.edge_linear(edge_features)
        edge_output = edge_hidden + self.edge_ffn(self.norm(edge_hidden))

        # Aggregation
        node_output = self.aggregate(edge_output, inputs.neighbor_mask)

        # Update inputs
        return node_output, edge_output
