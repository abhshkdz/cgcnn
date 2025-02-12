from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import Literal


@dataclass
class GlobalConfigs:
    regress_forces: bool
    direct_forces: bool
    hidden_size: int  # divisible by 2 and num_heads
    batch_size: int
    activation: Literal[
        "squared_relu", "gelu", "leaky_relu", "relu", "smelu", "star_relu"
    ]
    use_compile: bool = True
    use_padding: bool = True


@dataclass
class MolecularGraphConfigs:
    use_pbc: bool
    use_pbc_single: bool
    otf_graph: bool
    max_neighbors: int
    max_radius: float
    max_num_elements: int
    max_num_nodes_per_batch: int
    enforce_max_neighbors_strictly: bool
    distance_function: Literal["gaussian", "sigmoid", "linearsigmoid", "silu"]


@dataclass
class GraphNeuralNetworksConfigs:
    num_layers: int
    atom_embedding_size: int
    node_direction_embedding_size: int
    node_direction_expansion_size: int
    edge_distance_expansion_size: int
    edge_distance_embedding_size: int
    atten_name: Literal[
        "math",
        "memory_efficient",
        "flash",
    ]
    atten_num_heads: int
    readout_hidden_layer_multiplier: int
    output_hidden_layer_multiplier: int
    ffn_hidden_layer_multiplier: int
    use_angle_embedding: bool = True
    use_equivariant_force: bool = False
    energy_reduce: Literal["sum", "mean"] = "mean"


@dataclass
class RegularizationConfigs:
    mlp_dropout: float
    atten_dropout: float
    stochastic_depth_prob: float
    normalization: Literal["layernorm", "rmsnorm", "skip"]


@dataclass
class EScAIPConfigs:
    global_cfg: GlobalConfigs
    molecular_graph_cfg: MolecularGraphConfigs
    gnn_cfg: GraphNeuralNetworksConfigs
    reg_cfg: RegularizationConfigs


def resolve_type_hint(cls, field):
    """Resolves forward reference type hints from string to actual class objects."""
    if isinstance(field.type, str):
        resolved_type = getattr(cls, field.type, None)
        if resolved_type is None:
            resolved_type = globals().get(field.type, None)  # Try global scope
        if resolved_type is None:
            return field.type  # Fallback to string if not found
        return resolved_type
    return field.type


def init_configs(cls, kwargs):
    """
    Initialize a dataclass with the given kwargs.
    """
    init_kwargs = {}
    for field in fields(cls):
        field_name = field.name
        field_type = resolve_type_hint(cls, field)  # Resolve type

        if is_dataclass(field_type):  # Handle nested dataclass
            init_kwargs[field.name] = init_configs(field_type, kwargs)
        elif field_name in kwargs:  # Direct assignment
            init_kwargs[field_name] = kwargs[field_name]
        elif field.default is not MISSING:  # Assign default if available
            init_kwargs[field_name] = field.default
        else:
            raise ValueError(
                f"Missing required configuration parameter: '{field_name}' in '{cls.__name__}'"
            )

    return cls(**init_kwargs)
