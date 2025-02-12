"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch_geometric.nn import radius_graph

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import (
    compute_neighbors,
    get_pbc_distances,
    load_model_and_weights_from_checkpoint,
    radius_graph_pbc,
)

if TYPE_CHECKING:
    from torch_geometric.data import Batch


@dataclass
class GraphData:
    """Class to keep graph attributes nicely packaged."""

    edge_index: torch.Tensor
    edge_distance: torch.Tensor
    edge_distance_vec: torch.Tensor
    cell_offsets: torch.Tensor
    offset_distances: torch.Tensor
    neighbors: torch.Tensor
    batch_full: torch.Tensor  # used for GP functionality
    atomic_numbers_full: torch.Tensor  # used for GP functionality
    node_offset: int = 0  # used for GP functionality


class GraphModelMixin:
    """Mixin Model class implementing some general convenience properties and methods."""

    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
        enforce_max_neighbors_strictly=None,
        use_pbc_single=False,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        use_pbc_single = use_pbc_single or self.use_pbc_single
        otf_graph = otf_graph or self.otf_graph

        if enforce_max_neighbors_strictly is None:
            enforce_max_neighbors_strictly = getattr(
                self, "enforce_max_neighbors_strictly", True
            )

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                if use_pbc_single:
                    (
                        edge_index_per_system,
                        cell_offsets_per_system,
                        neighbors_per_system,
                    ) = list(
                        zip(
                            *[
                                radius_graph_pbc(
                                    data[idx],
                                    cutoff,
                                    max_neighbors,
                                    enforce_max_neighbors_strictly,
                                )
                                for idx in range(len(data))
                            ]
                        )
                    )

                    # atom indexs in the edge_index need to be offset
                    atom_index_offset = data.natoms.cumsum(dim=0).roll(1)
                    atom_index_offset[0] = 0
                    edge_index = torch.hstack(
                        [
                            edge_index_per_system[idx] + atom_index_offset[idx]
                            for idx in range(len(data))
                        ]
                    )
                    cell_offsets = torch.vstack(cell_offsets_per_system)
                    neighbors = torch.hstack(neighbors_per_system)
                else:
                    ## TODO this is the original call, but blows up with memory
                    ## using two different samples
                    ## sid='mp-675045-mp-675045-0-7' (MPTRAJ)
                    ## sid='75396' (OC22)
                    edge_index, cell_offsets, neighbors = radius_graph_pbc(
                        data,
                        cutoff,
                        max_neighbors,
                        enforce_max_neighbors_strictly,
                    )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return GraphData(
            edge_index=edge_index,
            edge_distance=edge_dist,
            edge_distance_vec=distance_vec,
            cell_offsets=cell_offsets,
            offset_distances=cell_offset_distances,
            neighbors=neighbors,
            node_offset=0,
            batch_full=data.batch,
            atomic_numbers_full=data.atomic_numbers.long(),
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> list:
        """Returns a list of parameters with no weight decay."""
        no_wd_list = []
        for name, _ in self.named_parameters():
            if "embedding" in name or "frequencies" in name or "bias" in name:
                no_wd_list.append(name)
        return no_wd_list


class HeadInterface(metaclass=ABCMeta):
    @property
    def use_amp(self):
        return False

    @abstractmethod
    def forward(
        self, data: Batch, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Head forward.

        Arguments
        ---------
        data: DataBatch
            Atomic systems as input
        emb: dict[str->torch.Tensor]
            Embeddings of the input as generated by the backbone

        Returns
        -------
        outputs: dict[str->torch.Tensor]
            Return one or more targets generated by this head
        """
        return


class BackboneInterface(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, data: Batch) -> dict[str, torch.Tensor]:
        """Backbone forward.

        Arguments
        ---------
        data: DataBatch
            Atomic systems as input

        Returns
        -------
        embedding: dict[str->torch.Tensor]
            Return backbone embeddings for the given input
        """
        return


@registry.register_model("hydra")
class HydraModel(nn.Module, GraphModelMixin):
    def __init__(
        self,
        backbone: dict | None = None,
        heads: dict | None = None,
        finetune_config: dict | None = None,
        otf_graph: bool = True,
        pass_through_head_outputs: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.device = None
        self.otf_graph = otf_graph
        # This is required for hydras with models that have multiple outputs per head, since we will deprecate
        # the old config system at some point, this will prevent the need to make major modifications to the trainer
        # because they all expect the name of the outputs directly instead of the head_name.property_name
        self.pass_through_head_outputs = pass_through_head_outputs

        # if finetune_config is provided, then attempt to load the model from the given finetune checkpoint
        starting_model = None
        if finetune_config is not None:
            # Make it hard to sneak more fields into finetuneconfig
            assert (
                len(set(finetune_config.keys()) - {"starting_checkpoint", "override"})
                == 0
            )
            starting_model: HydraModel = load_model_and_weights_from_checkpoint(
                finetune_config["starting_checkpoint"]
            )
            logging.info(
                f"Found and loaded fine-tuning checkpoint: {finetune_config['starting_checkpoint']} (Note we are NOT loading the training state from this checkpoint, only parts of the model and weights)"
            )
            assert isinstance(
                starting_model, HydraModel
            ), "Can only finetune starting from other hydra models!"
            # TODO this is a bit hacky to overrride attrs in the backbone
            if "override" in finetune_config:
                for key, value in finetune_config["override"].items():
                    setattr(starting_model.backbone, key, value)

        if backbone is not None:
            backbone = copy.deepcopy(backbone)
            backbone_model_name = backbone.pop("model")
            self.backbone: BackboneInterface = registry.get_model_class(
                backbone_model_name
            )(
                **backbone,
            )
        elif starting_model is not None:
            self.backbone = starting_model.backbone
            logging.info(
                f"User did not specify a backbone, using the backbone from the starting checkpoint {self.backbone}"
            )
        else:
            raise RuntimeError(
                "Backbone not specified and not found in the starting checkpoint"
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if heads is not None:
            heads = copy.deepcopy(heads)
            # Iterate through outputs_cfg and create heads
            self.output_heads: dict[str, HeadInterface] = {}

            head_names_sorted = sorted(heads.keys())
            assert len(set(head_names_sorted)) == len(
                head_names_sorted
            ), "Head names must be unique!"
            for head_name in head_names_sorted:
                head_config = heads[head_name]
                if "module" not in head_config:
                    raise ValueError(
                        f"{head_name} head does not specify module to use for the head"
                    )

                module_name = head_config.pop("module")
                self.output_heads[head_name] = registry.get_model_class(module_name)(
                    self.backbone,
                    **head_config,
                )

            self.output_heads = torch.nn.ModuleDict(self.output_heads)
        elif starting_model is not None:
            self.output_heads = starting_model.output_heads
            logging.info(
                f"User did not specify heads, using the output heads from the starting checkpoint {self.output_heads}"
            )
        else:
            raise RuntimeError(
                "Heads not specified and not found in the starting checkpoint"
            )

    def forward(self, data: Batch):
        # lazily get device from input to use with amp, at least one input must be a tensor to figure out it's device
        if not self.device:
            device_from_tensors = {
                x.device.type for x in data.values() if isinstance(x, torch.Tensor)
            }
            assert (
                len(device_from_tensors) == 1
            ), f"all inputs must be on the same device, found the following devices {device_from_tensors}"
            self.device = device_from_tensors.pop()

        emb = self.backbone(data)
        # Predict all output properties for all structures in the batch for now.
        out = {}
        for k in self.output_heads:
            with torch.autocast(
                device_type=self.device, enabled=self.output_heads[k].use_amp
            ):
                if self.pass_through_head_outputs:
                    out.update(self.output_heads[k](data, emb))
                else:
                    out[k] = self.output_heads[k](data, emb)

        return out
