from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from fairchem.core.models.escaip.configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        RegularizationConfigs,
    )
from fairchem.core.models.escaip.utils.nn_utils import (
    get_feedforward,
    get_normalization_layer,
)


class ReadoutBlock(nn.Module):
    """
    Readout from each graph attention block for energy and force output
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        self.regress_forces = global_cfg.regress_forces
        self.direct_forces = global_cfg.direct_forces

        # energy read out
        self.energy_ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.readout_hidden_layer_multiplier,
            dropout=reg_cfg.mlp_dropout,
            bias=True,
        )
        self.energy_norm = get_normalization_layer(
            reg_cfg.normalization, is_graph=False
        )(global_cfg.hidden_size)

        # forces read out
        if self.regress_forces and self.direct_forces:
            self.force_ffn = get_feedforward(
                hidden_dim=global_cfg.hidden_size,
                activation=global_cfg.activation,
                hidden_layer_multiplier=gnn_cfg.readout_hidden_layer_multiplier,
                dropout=reg_cfg.mlp_dropout,
                bias=True,
            )
            self.force_norm = get_normalization_layer(
                reg_cfg.normalization, is_graph=False
            )(global_cfg.hidden_size)
        else:
            self.force_ffn = nn.Identity()
            self.force_norm = nn.Identity()

    def forward(self, node_features, edge_features):
        """
        Output: Node Readout (N, H); Edge Readout (N, max_nei, H)
        """
        energy_readout = node_features + self.energy_ffn(
            self.energy_norm(node_features)
        )
        force_readout = edge_features + self.force_ffn(self.force_norm(edge_features))

        return energy_readout, force_readout
