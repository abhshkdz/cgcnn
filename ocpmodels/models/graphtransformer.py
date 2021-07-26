"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Union
from argparse import Namespace
import numpy
import os
import scipy.stats as stats
import torch
from torch import nn as nn
from torch.nn import LayerNorm, functional as F
from torch_geometric.nn import radius_graph
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_scatter import scatter, scatter_mean
from torch_geometric.utils import degree

from ocpmodels.models.base import BaseModel
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

@registry.register_model("graphtransformer")
class GraphTransformer(BaseModel):
    r"""Using graph transformer architecture from GROVER model from the
    `"Self-Supervised Graph Transformer on Large-Scale Molecular Data" <https://arxiv.org/abs/2007.02835>_ paper.
    Implementation adapted from: https://github.com/tencent-ailab/grover.

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        num_gaussians(int, optional): Number of gaussian distributions for distance expansion (from SchNet)
            (Default: :obj:`50`)
        hidden_size (int, optional): Dimensionality of hidden layers. The actual dimension is hidden_size * 100.
            (default: :obj:`128`)
        bias (bool, optional): If set to :obj:`True` add bias to linear layers.
            (default: :obj:`False`)
        depth (int, optional): Number of message passing steps.
            (default: :obj:`3`)
        dropout (float, optional): Dropout probability.
            (default: :obj:`0.0`)
        activation(string, optional): Activation function, choices: ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'.
            (default: :obj:`ReLU`)
        weight_decay (float, optional): Weight decay.
            (default: :obj:`0.0`)
        num_attn_head (int, optional): Number of attention heads per MTBlock.
            (default: :obj:`4`)
        num_mt_block (int, optional): Number of MTBlocks.
            (default: :obj:`1`)
        res_connection (bool, optional): If set to :obj:`True`, enables the skip-connection in MTBlock.
            (default: :obj:`False`)
        embedding_output_type (string, optional): Output type for GTransEncoder, choices: 'atom', 'bond', 'both'.
            (default: :obj:`both`)
        ffn_hidden_Size (int, optional): Hidden dimension for higher-capacity FFN (defaults to hidden_size).
            (default: :obj:`None`)
        ffn_num_layers (int, optional): Number of layers in FFN after MPN encoding.
            (default: :obj:`2`)
        self_attention (bool, optional): If set to :obj:`True`, use self-attention layer, otherwise use mean aggregation layer.
            (default: :obj:`False`)
        attn_hidden (int, optional): Self attention layer hidden layer size.
            (default: :obj:`4`)
        attn_out (int, optional): Self attention layer output feature size.
            (default: :obj:`128`)
        undirected (bool, optional): If set to :obj:`True`, specify that the molecular graph is undirected.
            (default: :obj:`True`)
        dense (bool, optional): If set to :obj:`True`, use dense connection in MessagePassing layer.
            (default: :obj:`True`)
        features_only (bool, optional): If set to :obj:`True`, use only the additional features in an FFN, no graph network
            (default: :obj:`False`)
        cuda (bool, optional): If set to :obj:`True`, use CUDA acceleration.
            (default: :obj:`True`)
        debug (bool, optional): If set to :obj:`True`, show debugging info and test convert_input against slower iterative approach
            (default: :obj:`False`)
        dynamic_depth (string, optional): Enables dynamic depth message passing. Possible choices: "none", "uniform" and "truncnorm"
            (default: :obj:`none`)
        pred_ffn_intermediate_dim (int, optional): Experimental intermediate layer for energy/force prediction ffn: hidden size -> ffn_hidden_size -> intermediate_dim -> output_size
            (default: :obj:`None`)
        pred_ffn_num_layers (int, optional): Number of layers for the ffn used to predict forces & energies
            (default: :obj:`2`)

    """
    def __init__(
        self,
        num_atoms,              # OCP: Number of atoms (not used)
        bond_feat_dim,          # OCP: Bond feature dimension (not used)
        num_targets,            # OCP: Number of targets to predict
        use_pbc=True,           # OCP: use periodic boundary conditions
        regress_forces=True,    # OCP: Regress forces as well as energy
        otf_graph=False,        # OCP: Use on-the-fly graph calculation
        cutoff=10.0,            # OCP: cutoff distance (angstroms)
        num_gaussians=50,       # OCP: number of gaussian distributions to expand distances
        hidden_size=3, # Dimensionality of hidden layers. The actual dimension is hidden_size * 100.
        bias=False, # Whether to add bias to linear layers
        depth=3, # Number of message passing steps
        dropout=0.0, # Dropout probability
        activation='ReLU', # Choices: ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'
        weight_decay=0.0, # Weight decay
        num_attn_head=4, # Number of attention heads per MTBlock
        num_mt_block=1, # Number of MTBlocks
        res_connection=False, # enables the skip-connection in MTBlock.
        embedding_output_type='both', # Choices: 'atom', 'bond', 'both'
        ffn_hidden_size=None, # Hidden dim for higher-capacity FFN (defaults to hidden_size)
        ffn_num_layers=2, # Number of layers in FFN after MPN encoding
        self_attention=False, # Use self attention layer, otherwise use mean aggregation layer
        attn_hidden=4, # Self attention layer hidden layer size
        attn_out=128, # Self attention layer output feature size
        undirected=True, # Specify whether or not molecular graph is undirected (OCP is undirected)
        dense=True, # Use dense connection in MessagePassing layer
        features_only=False, # Use only the additional features in an FFN, no graph network
        cuda=True, # Use CUDA acceleration
        debug=False, # Print debugging info
        dynamic_depth="none", # Enables dynamic depth message passing. Possible choices: "none", "uniform" and "truncnorm"
        pred_ffn_intermediate_dim=None, # Intermediate dimension for the ffn used to predict forces & energies
        pred_ffn_num_layers=2 # Number of layers for the ffn used to predict forces & energies
    ):
        # OCP parameters
        self.num_targets = num_targets
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.cutoff = cutoff
        self.num_gaussians = num_gaussians

        # GraphTransformer parameters
        args = Namespace()
        args.hidden_size = hidden_size
        args.bias = bias
        args.depth = depth
        args.dropout = dropout
        args.activation = activation
        args.weight_decay = weight_decay
        args.num_attn_head = num_attn_head
        args.num_mt_block = num_mt_block
        args.res_connection = res_connection
        args.embedding_output_type = embedding_output_type
        args.ffn_hidden_size = ffn_hidden_size if ffn_hidden_size is not None else hidden_size # Defaults to hidden size if None
        args.ffn_num_layers = ffn_num_layers
        args.self_attention = self_attention
        args.attn_hidden = attn_hidden
        args.attn_out = attn_out
        args.cuda = cuda
        args.undirected = undirected
        args.dense = dense
        args.features_only = features_only
        args.features_dim = 0 # Temproary for troubleshooting
        args.features_size = 0 # Temporary for troubleshooting
        args.debug = debug
        args.dynamic_depth = dynamic_depth
        args.pred_ffn_intermediate_dim = pred_ffn_intermediate_dim
        args.pred_ffn_num_layers = pred_ffn_num_layers

        self.args = args # Hack to call in forward pass

        super(GraphTransformer, self).__init__()
        self.distance_expansion = GaussianSmearing(0.0, cutoff, self.num_gaussians)

        self.edge_fdim = self.num_gaussians if not use_pbc else self.num_gaussians + 3 # Extra 3 dimensions for dist vectors
        self.node_fdim = 4 # Node features are of dimension 4: atomic number, pos_x, pos_y, pos_z

        self.encoders = GTransEncoder(args,
                                      hidden_size=args.hidden_size,
                                      edge_fdim=self.edge_fdim,
                                      node_fdim=self.node_fdim,
                                      dropout=args.dropout,
                                      activation=args.activation,
                                      num_mt_block=args.num_mt_block,
                                      num_attn_head=args.num_attn_head,
                                      atom_emb_output=args.embedding_output_type,
                                      bias=args.bias,
                                      cuda=args.cuda,
                                      res_connection=args.res_connection)

        # Separate feed forward layers from atom embeddings to calculate energy & forces (3D)
        self.energy_atom_from_atom_ffn = self.create_ffn(args, output_dim=1, intermediate_dim=args.pred_ffn_intermediate_dim, num_layers=args.pred_ffn_num_layers)
        self.energy_atom_from_bond_ffn = self.create_ffn(args, output_dim=1, intermediate_dim=args.pred_ffn_intermediate_dim, num_layers=args.pred_ffn_num_layers)
        self.energy_atom_from_bond_aggregated_ffn = self.create_ffn(args, output_dim=1, intermediate_dim=args.pred_ffn_intermediate_dim, num_layers=args.pred_ffn_num_layers)

        self.forces_atom_from_atom_ffn = self.create_ffn(args, output_dim=3, intermediate_dim=args.pred_ffn_intermediate_dim, num_layers=args.pred_ffn_num_layers)
        self.forces_atom_from_bond_ffn = self.create_ffn(args, output_dim=3, intermediate_dim=args.pred_ffn_intermediate_dim, num_layers=args.pred_ffn_num_layers)
        self.forces_atom_from_bond__aggregated_ffn = self.create_ffn(args, output_dim=3, intermediate_dim=args.pred_ffn_intermediate_dim, num_layers=args.pred_ffn_num_layers)


    # FFNs for GTransformer output embeddings
    def create_ffn(self, args: Namespace, output_dim, intermediate_dim, num_layers):
        """
        Creates the feed-forward network for the model.
        """
        # Hard code output dimension
        args.output_size = output_dim

        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                first_linear_dim += args.features_dim
            else:
                first_linear_dim = args.hidden_size + args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            if intermediate_dim:
                if(num_layers == 3):
                    ffn.extend([
                        activation,
                        dropout,
                        nn.Linear(args.ffn_hidden_size, intermediate_dim)
                    ])
                    ffn.extend([
                        activation,
                        dropout,
                        nn.Linear(intermediate_dim, args.output_size),
                    ])
                else: # Only allows 2 or 3 layers, otherwise too much learning is done by ffn and not GTransEncoder
                    ffn.extend([
                        activation,
                        dropout,
                        nn.Linear(args.ffn_hidden_size, args.output_size)
                    ])
            else:
                for _ in range(args.ffn_num_layers - 2):
                    ffn.extend([
                        activation,
                        dropout,
                        nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                    ])
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.output_size),
                ])

        # Create FFN model
        return nn.Sequential(*ffn)

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        args = self.args
        pos = data.pos
        batch = data.batch

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50, data.pos.device
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
                return_distance_vec=True
            )

            data.edge_index = out["edge_index"]
            distances = out["distances"]
            edge_vec = out["distance_vec"]
        else:
            data.edge_index = radius_graph(
                data.pos, r=self.cutoff, batch=data.batch
            )
            row, col = data.edge_index
            distances = (pos[row] - pos[col]).norm(dim=-1)

        data.edge_attr = self.distance_expansion(distances) # Features will be of dimensions [num_edges, num_gaussians]
        if self.use_pbc:
            data.edge_attr = torch.cat((data.edge_attr, edge_vec), dim=1) # Append difference in position (x, y, z) to edge attr

        # Convert to format from PyTorch Geometric to explicit mappings format used by GROVER
        converted_input = convert_input(args, data)

        # GTransEncoder
        output = self.encoders(converted_input)

        embeddings = {}
        if args.embedding_output_type == 'atom':
            embeddings = {"atom_from_atom": output[0], "atom_from_bond": output[1],
                          "bond_from_atom": None, "bond_from_bond": None}  # atom_from_atom, atom_from_bond
        elif args.embedding_output_type == 'bond':
            embeddings = {"atom_from_atom": None, "atom_from_bond": None,
                          "bond_from_atom": output[0], "bond_from_bond": output[1]}  # bond_from_atom, bond_from_bond
        elif args.embedding_output_type == "both":
            embeddings = {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
                          "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}

        if args.embedding_output_type == 'atom' or args.embedding_output_type == 'both':
            # Get atom embeddings (handles different embedding_output_type)
            atom_from_bond_output = embeddings["atom_from_bond"]
            atom_from_atom_output = embeddings["atom_from_atom"]

            # Use info from data batch to only sum up energy per system
            batch = data.batch

            # Energy calculated by passing atom embeddings through feed-forward layer, summing over atoms in the system
            atom_ffn_energy = self.energy_atom_from_atom_ffn(atom_from_atom_output) # output is 1 dimensional per atom
            bond_ffn_energy = self.energy_atom_from_bond_ffn(atom_from_bond_output) # output is 1 dimensional per atom
            atom_energy = atom_ffn_energy.sum(dim=0) if batch is None else scatter(atom_ffn_energy, batch, dim=0)
            bond_energy = bond_ffn_energy.sum(dim=0) if batch is None else scatter(bond_ffn_energy, batch, dim=0)

            # Per-atom forces calculated through feed-forward layer from embeddings
            atom_forces = self.forces_atom_from_atom_ffn(atom_from_atom_output) # output is 3 dimensional force prediction
            bond_forces = self.forces_atom_from_bond_ffn(atom_from_bond_output) # output is 3 dimensional force prediction

            if args.embedding_output_type == 'atom':
                # Using atom embeddings only, average predictions from two different atom embeddings (from atom and from bond)
                energy = (atom_energy + bond_energy) / 2
                forces = (atom_forces + bond_forces) / 2

        # If we have bond embeddings, use these to calculate new node embeddings by summing incoming edge embeddings
        if args.embedding_output_type == 'bond' or args.embedding_output_type == 'both':
            bond_embeddings = (embeddings["bond_from_atom"] + embeddings[
                "bond_from_bond"]) / 2  # avg two bond embeddings
            _, _, a2b, _, _, _, _, _ = converted_input # Get a2b from converted input

            new_atom_embeddings = scatter_mean(bond_embeddings, data.edge_index[1], dim=0)  # Sum up incoming bond embeddings for every atom

            # Calculate energy and forces with these new embeddings
            new_bond_ffn_energy = self.energy_atom_from_bond_aggregated_ffn(new_atom_embeddings)
            new_bond_energy = new_bond_ffn_energy.sum(dim=0) if batch is None else scatter(new_bond_ffn_energy, batch,
                                                                                           dim=0)
            new_bond_forces = self.forces_atom_from_bond__aggregated_ffn(new_atom_embeddings)

            if args.embedding_output_type == 'both':
                # Average predictionss from new atom embeddings (from edges), with other preds from atom embeddings
                energy = (atom_energy + bond_energy + new_bond_energy) / 3
                forces = (atom_forces + bond_forces + new_bond_forces) / 3
            else:
                # Using edge embedding only
                energy = new_bond_energy
                forces = new_bond_forces

        return energy, forces

    # From SchNet/dimenet: calculate forces through gradients--we will calculate them explicitly with transformer for now
    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy, forces = self._forward(data)

        if self.regress_forces:
            # forces = -1 * (
            #     torch.autograd.grad(
            #         energy,
            #         data.pos,
            #         grad_outputs=torch.ones_like(energy),
            #         create_graph=True,
            #     )[0]
            # )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# Helper function to convert from PyTorch Geometric format to explicit mappings used by GROVER:
def convert_input(args, data):
    """
        :param data: data as PyTorch geometric object
        :param f_atoms: the atom features, num_atoms * atom_dim
        :param f_bonds: the bond features, num_bonds * bond_dim
        :param a2b: mapping from atom index to incoming bond indices.
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :param a_scope: A list of tuples indicating the start and end atom indices for each molecule (system/batch in ocp)
        :param b_scope: A list of tuples indicating the start and end bond indices for each molecule (system/batch in ocp)
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :return: batch = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
    """
    # Per atom features: (atomic_number, pos_x, pos_y, pos_z)
    f_atoms = torch.stack((data.atomic_numbers.long(), data.pos[:, 0], data.pos[:, 1], data.pos[:, 2]), 1)
    # Per edge features (calculated by atomic distances in model forward pass)
    f_bonds = data.edge_attr

    # Lists of tuples indicating the start and end of each molecule (system/batch in ocp)
    a_scope = []
    b_scope = []

    # Set a_scope by looping through the total number of systems in the batch (numbered 0, 1, 2, ... n)
    for i in range(len(data.natoms)):
        if (i == 0):
            start_index = 0
        system_size = int(data.natoms[i])  # Number of atoms in this system
        end_index = start_index + system_size - 1  # End index is 1 before the start of the next system
        a_scope.append((int(start_index), int(end_index)))
        start_index = start_index + system_size  # Update start index to move to the next system

    # Set b_scope using data.neighbors, which categorizes eddges by system (neighbors[i] is number of edges in system i)
    for i in range(len(data.neighbors)):
        if(i == 0):
            start_index = 0
        system_size = int(data.neighbors[i]) # Number of edges in this system
        end_index = start_index + system_size - 1 # End index is 1 before the start of the next system
        b_scope.append((start_index, end_index))
        start_index = start_index + system_size # Update start index to move to the next system

    num_atoms_total = int(torch.sum(data.natoms))

    # Calculate atom to neighboring atom mappings (a2a)
    trans = data.edge_index.T
    sorted_index = trans[trans[:, 0].sort()[1]]  # Sort by column zero (from_node), index based off
    sorted_index = sorted_index.T

    out, counts = torch.unique(sorted_index[0], return_counts=True)  # counts for each atom
    if len(out) - 1 != out[-1]: # If they are the correct length then we don't need to add 0's
        ascending = torch.arange(out[-1] + 1).cuda() # Ascending order list [0, 1, ... n] where n is last elem in out
        missing = [ elem not in out for elem in ascending ] # Missing elements in out
        for elem in numpy.nonzero(missing)[0]:
            i = int(elem)
            counts = torch.cat((counts[:i], torch.tensor((0,)).cuda(), counts[i:])) # Add zero to count
    a2a2 = sorted_index[1].split(counts.tolist())  # Index into to_bonds with these indices
    a2a2 = torch.nn.utils.rnn.pad_sequence(list(a2a2), batch_first=True,
                                           padding_value=0) # Pad with zeros into a tensor
    a2a2 = a2a2.cpu()

    if len(a2a2) < num_atoms_total:
        a2a2 = torch.cat((a2a2, torch.zeros((num_atoms_total - len(a2a2), a2a2.shape[1]))))  # Add extra entry of zeros to match dimensions

    # Calculate outgoing bond to atom mappings (b2a)
    b2a2 = data.edge_index[0].type(torch.LongTensor).cpu()

    # Calculate atom to incoming bond mappings (a2b)
    out, counts = torch.unique(data.edge_index[1], return_counts=True)
    if len(out) - 1 != out[-1]: # If they are the correct length then we don't need to add 0's
        ascending = torch.arange(out[-1] + 1).cuda() # Ascending order list [0, 1, ... n] where n is last elem in out
        missing = [ elem not in out for elem in ascending ] # Missing elements in out
        for elem in numpy.nonzero(missing)[0]:
            i = int(elem)
            counts = torch.cat((counts[:i], torch.tensor((0,)).cuda(), counts[i:])) # Add zero to count

    a2b2 = torch.split(torch.arange(len(data.edge_index[1])), tuple(counts))
    a2b2 = torch.nn.utils.rnn.pad_sequence(a2b2, batch_first=True, padding_value=0)  # Pad with zeros into a tensor
    a2b2 = a2b2.cpu()
    if len(a2b2) < num_atoms_total:
        a2b2 = torch.cat((a2b2, torch.zeros((num_atoms_total - len(a2b2), a2b2.shape[1]))))  # Add extra entry of zeros to match dimensions

    a2a2 = a2a2.type(torch.LongTensor)
    a2b2 = a2b2.type(torch.LongTensor)

    # If we need to debug, test against guaranteed (but inefficient) way to do it
    if args.debug:
        # OLD CODE FOR CALCULATING MAPPINGS
        num_atoms_total = int(torch.sum(data.natoms))
        a2a = [[] for j in range(num_atoms_total)]  # List of lists - Dynamically append neighbors for a given atom
        a2b = [[] for j in range(num_atoms_total)]  # List of lists - Dynamically append edges for a given atom
        b2a = torch.zeros((data.edge_index.shape[1],)).long() # (num_edges, ) - One originating atom per edge
        b2revb = torch.zeros((data.edge_index.shape[1],)).long()  # (num_edges, ) - One reverse bond per bond
        rev_edges = {}  # Dict of lists for each (from_atom, to_atom) pair, saving edge numbers

        # Loop through every edge in the graph
        for i in range(data.edge_index.shape[1]):
            from_atom = int(data.edge_index[0][i])
            to_atom = int(data.edge_index[1][i])
            a2a[from_atom].append(to_atom)  # Mark b as neighbor of a
            a2b[to_atom].append(i)  # Mark bond i as incoming bond to atom a
            b2a[i] = from_atom  # Mark a as atom where bond i is originating
            key = frozenset({to_atom, from_atom})
            if (key not in rev_edges):  # If the edge from these two atoms has not been seen yet
                rev_edges[key] = []  # Declare it as a list (so we can keep track of the edge numbers)
            rev_edges[key].append(i)  # Append the edge number to the list

        # Iterate through and set b2revb with reverse bonds
        for atoms, edges in rev_edges.items():
            if(len(edges) == 2):
                b2revb[edges[0]] = edges[1]
                b2revb[edges[1]] = edges[0]
            elif(len(edges) == 4): # In the case of duplicate edges, they are grouped together in this order
                b2revb[edges[0]] = edges[2]
                b2revb[edges[2]] = edges[0]
                b2revb[edges[1]] = edges[3]
                b2revb[edges[3]] = edges[1]
            elif(len(edges) == 1):
                args.undirected = False

        # Convert list of lists for a2a and a2b into tensor: (num_nodes, max_edges)
        # Trim length to max number of edges seen in the data (should be capped by 50 but not always in practice)
        a2a_pad = len(max(a2a, key=len))
        a2b_pad = len(max(a2b, key=len))
        # -1 is not a valid atom or edge index so we pad with this
        a2a = torch.tensor([i + [0] * (a2a_pad - len(i)) for i in a2a])
        a2b = torch.tensor([i + [0] * (a2b_pad - len(i)) for i in a2b])
    
        a2a = a2a.cpu()
        a2b = a2b.cpu()
        b2a = b2a.cpu()

        # COMPARE IMPLEMENTATIONS
        if not torch.equal(b2a, b2a2):
            print("a2b and a2b2 equal? ", torch.equal(a2b, a2b2))
            print("b2a and b2a2 equal? ", torch.equal(b2a, b2a2))
            print("a2a and a2a2 equal? ", torch.equal(a2a, a2a2))
            print("a2a and a2a2 equal counts: ",
                  torch.equal(torch.unique(a2a, return_counts=True)[1], torch.unique(a2a2, return_counts=True)[1]))
            print("\nb2a shape: ", b2a.shape)
            print("b2a2 shape: ", b2a2.shape)
            path = "logs/error_dumps/b2a" + str(list(b2a.shape)).strip("[]")
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(b2a, path + "/b2a.pt")
            torch.save(b2a2, path + "/b2a2.pt")
            torch.save(data.edge_index, path + "/edge_index.pt")
            raise Exception("Incorrect b2a calculation")

        if not torch.equal(a2b, a2b2):
            print("a2b and a2b2 equal? ", torch.equal(a2b, a2b2))
            print("b2a and b2a2 equal? ", torch.equal(b2a, b2a2))
            print("a2a and a2a2 equal? ", torch.equal(a2a, a2a2))
            print("a2a and a2a2 equal counts: ",
                  torch.equal(torch.unique(a2a, return_counts=True)[1], torch.unique(a2a2, return_counts=True)[1]))
            print("\na2b shape: ", a2b.shape)
            print("a2b2 shape: ", a2b2.shape)
            path = "logs/error_dumps/a2b" + str(list(b2a.shape)).strip("[]").replace(",", "-").replace(" ", "")
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(a2b, path + "/a2b.pt")
            torch.save(a2b2, path + "/a2b2.pt")
            torch.save(data.edge_index, path + "/edge_index.pt")
            raise Exception("Incorrect a2b calculation")

        if not torch.equal(a2a, a2a2) or not torch.equal(torch.unique(a2a, return_counts=True)[1], torch.unique(a2a2, return_counts=True)[1]):
            print("a2b and a2b2 equal? ", torch.equal(a2b, a2b2))
            print("b2a and b2a2 equal? ", torch.equal(b2a, b2a2))
            print("a2a and a2a2 equal? ", torch.equal(a2a, a2a2))
            print("a2a and a2a2 equal counts: ",
                  torch.equal(torch.unique(a2a, return_counts=True)[1], torch.unique(a2a2, return_counts=True)[1]))
            print("\na2a count: ", torch.unique(a2a, return_counts=True)[1])
            print("a2a2 count: ", torch.unique(a2a2, return_counts=True)[1])
            print("a2a shape: ", a2a.shape)
            print("a2a2 shape: ", a2a2.shape)
            path = "logs/error_dumps/a2a" + str(list(b2a.shape)).strip("[]").replace(",", "-").replace(" ", "")
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(a2a, path + "/a2a.pt")
            torch.save(a2a2, path + "/a2a2.pt")
            torch.save(data.edge_index, path + "/edge_index.pt")
            raise Exception("Incorrect a2a calculation")

    # Use more efficient calculations of a2a/a2b/b2a
    a2b = a2b2
    a2a = a2a2
    b2a = b2a2

    # Set reverse bond mapping to all zeros, as our graph is mostly undirected
    b2revb = torch.zeros((data.edge_index.shape[1],)).long()

    batch = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
    return batch

# Classes below from GROVER implementation (https://github.com/tencent-ailab/grover)
class SelfAttention(nn.Module):
    """
       Self SelfAttention Layer
       Given $X\in \mathbb{R}^{n \times in_feature}$, the attention is calculated by: $a=Softmax(W_2tanh(W_1X))$, where
       $W_1 \in \mathbb{R}^{hidden \times in_feature}$, $W_2 \in \mathbb{R}^{out_feature \times hidden}$.
       The final output is: $out=aX$, which is unrelated with input $n$.
    """
    def __init__(self, *, hidden, in_feature, out_feature):
        """
        The init function.
        :param hidden: the hidden dimension, can be viewed as the number of experts.
        :param in_feature: the input feature dimension.
        :param out_feature: the output feature dimension.
        """
        super(SelfAttention, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden, in_feature))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(out_feature, hidden))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Use xavier_normal method to initialize parameters.
        """
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, X):
        """
        The forward function.
        :param X: The input feature map. $X \in \mathbb{R}^{n \times in_feature}$.
        :return: The final embeddings and attention matrix.
        """
        x = torch.tanh(torch.matmul(self.w1, X.transpose(1, 0)))
        x = torch.matmul(self.w2, x)
        attn = torch.nn.functional.softmax(x, dim=-1)
        x = torch.matmul(attn, X)
        return x, attn

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product SelfAttention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        """
        :param query:
        :param key:
        :param value:
        :param mask:
        :param dropout:
        :return:
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / numpy.math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    The multi-head attention module. Take in model size and number of heads.
    """
    def __init__(self, h, d_model, dropout=0.1, bias=False):
        """
        :param h:
        :param d_model:
        :param dropout:
        :param bias:
        """
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h  # number of heads
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)])  # why 3: query, key, value
        self.output_linear = nn.Linear(d_model, d_model, bias)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        :param query:
        :param key:
        :param value:
        :param mask:
        :return:
        """
        batch_size = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)

class Head(nn.Module):
    """
    One head for multi-headed attention.
    :return: (query, key, value)
    """
    def __init__(self, args, hidden_size, atom_messages=False):
        """
        Initialization.
        :param args: The argument.
        :param hidden_size: the dimension of hidden layer in Head.
        :param atom_messages: the MPNEncoder type.
        """
        super(Head, self).__init__()
        atom_fdim = hidden_size
        bond_fdim = hidden_size
        hidden_size = hidden_size
        self.atom_messages = atom_messages
        if self.atom_messages:
            init_message_dim = atom_fdim
            attached_fea_dim = bond_fdim
        else:
            init_message_dim = bond_fdim
            attached_fea_dim = atom_fdim
        # Here we use the message passing network as query, key and value.
        self.mpn_q = MPNEncoder(args=args,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=args.bias,
                                depth=args.depth,
                                dropout=args.dropout,
                                undirected=args.undirected,
                                dense=args.dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm")
        self.mpn_k = MPNEncoder(args=args,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=args.bias,
                                depth=args.depth,
                                dropout=args.dropout,
                                undirected=args.undirected,
                                dense=args.dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm")
        self.mpn_v = MPNEncoder(args=args,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=args.bias,
                                depth=args.depth,
                                dropout=args.dropout,
                                undirected=args.undirected,
                                dense=args.dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm")

    def forward(self, f_atoms, f_bonds, a2b, a2a, b2a, b2revb):
        """
        The forward function.
        :param f_atoms: the atom features, num_atoms * atom_dim
        :param f_bonds: the bond features, num_bonds * bond_dim
        :param a2b: mapping from atom index to incoming bond indices.
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :return:
        """
        if self.atom_messages:
            init_messages = f_atoms
            init_attached_features = f_bonds
            a2nei = a2a
            a2attached = a2b
            b2a = b2a
            b2revb = b2revb
        else:
            init_messages = f_bonds
            init_attached_features = f_atoms
            a2nei = a2b
            a2attached = a2a
            b2a = b2a
            b2revb = b2revb
        q = self.mpn_q(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        k = self.mpn_k(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        v = self.mpn_v(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        return q, k, v

class MTBlock(nn.Module):
    """
    The Multi-headed attention block.
    """
    def __init__(self,
                 args,
                 num_attn_head,
                 input_dim,
                 hidden_size,
                 activation="ReLU",
                 dropout=0.0,
                 bias=True,
                 atom_messages=False,
                 cuda=True,
                 res_connection=False):
        """
        :param args: the arguments.
        :param num_attn_head: the number of attention head.
        :param input_dim: the input dimension.
        :param hidden_size: the hidden size of the model.
        :param activation: the activation function.
        :param dropout: the dropout ratio
        :param bias: if true: all linear layer contains bias term.
        :param atom_messages: the MPNEncoder type
        :param cuda: if true, the model run with GPU.
        :param res_connection: enables the skip-connection in MTBlock.
        """
        super(MTBlock, self).__init__()
        self.args = args
        self.atom_messages = atom_messages
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList()
        self.input_dim = input_dim
        self.cuda = cuda
        self.res_connection = res_connection
        self.act_func = get_activation_function(activation)
        self.dropout_layer = nn.Dropout(p=dropout)
        # Note: elementwise_affine has to be consistent with the pre-training phase
        self.layernorm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        self.W_i = nn.Linear(self.input_dim, self.hidden_size, bias=bias)
        self.attn = MultiHeadedAttention(h=num_attn_head,
                                         d_model=self.hidden_size,
                                         bias=bias,
                                         dropout=dropout)
        self.W_o = nn.Linear(self.hidden_size * num_attn_head, self.hidden_size, bias=bias)
        self.sublayer = SublayerConnection(self.hidden_size, dropout)
        for _ in range(num_attn_head):
            self.heads.append(Head(args, hidden_size=hidden_size, atom_messages=atom_messages))

    def forward(self, batch, features_batch=None):
        """
        :param batch: the graph batch generated by GroverCollator.
        :param features_batch: the additional features of molecules. (deprecated)
        :return:
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
        if self.atom_messages:
            # Only add linear transformation in the input feature.
            if f_atoms.shape[1] != self.hidden_size:
                f_atoms = self.W_i(f_atoms)
                f_atoms = self.dropout_layer(self.layernorm(self.act_func(f_atoms)))
        else:  # bond messages
            if f_bonds.shape[1] != self.hidden_size:
                f_bonds = self.W_i(f_bonds)
                f_bonds = self.dropout_layer(self.layernorm(self.act_func(f_bonds)))
        queries = []
        keys = []
        values = []
        for head in self.heads:
            q, k, v = head(f_atoms, f_bonds, a2b, a2a, b2a, b2revb)
            queries.append(q.unsqueeze(1))
            keys.append(k.unsqueeze(1))
            values.append(v.unsqueeze(1))
        queries = torch.cat(queries, dim=1)
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)
        x_out = self.attn(queries, keys, values)  # multi-headed attention
        x_out = x_out.view(x_out.shape[0], -1)
        x_out = self.W_o(x_out)
        x_in = None
        # support no residual connection in MTBlock.
        if self.res_connection:
            if self.atom_messages:
                x_in = f_atoms
            else:
                x_in = f_bonds
        if self.atom_messages:
            f_atoms = self.sublayer(x_in, x_out)
        else:
            f_bonds = self.sublayer(x_in, x_out)
        batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        features_batch = features_batch
        return batch, features_batch

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        """Initialization.
        :param size: the input dimension.
        :param dropout: the dropout ratio.
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, inputs, outputs):
        """Apply residual connection to any sublayer with the same size."""
        # return x + self.dropout(self.norm(x))
        if inputs is None:
            return self.dropout(self.norm(outputs))
        return inputs + self.dropout(self.norm(outputs))

class GTransEncoder(nn.Module):
    def __init__(self,
                 args,
                 hidden_size,
                 edge_fdim,
                 node_fdim,
                 dropout=0.0,
                 activation="ReLU",
                 num_mt_block=1,
                 num_attn_head=4,
                 atom_emb_output: Union[bool, str] = False,  # options: True, False, None, "atom", "bond", "both"
                 bias=False,
                 cuda=True,
                 res_connection=False):
        """
        :param args: the arguments.
        :param hidden_size: the hidden size of the model.
        :param edge_fdim: the dimension of additional feature for edge/bond.
        :param node_fdim: the dimension of additional feature for node/atom.
        :param dropout: the dropout ratio
        :param activation: the activation function
        :param num_mt_block: the number of mt block.
        :param num_attn_head: the number of attention head.
        :param atom_emb_output:  enable the output aggregation after message passing.
                                              atom_messages:      True                      False
        -False: no aggregating to atom. output size:     (num_atoms, hidden_size)    (num_bonds, hidden_size)
        -True:  aggregating to atom.    output size:     (num_atoms, hidden_size)    (num_atoms, hidden_size)
        -None:                         same as False
        -"atom":                       same as True
        -"bond": aggragating to bond.   output size:     (num_bonds, hidden_size)    (num_bonds, hidden_size)
        -"both": aggregating to atom&bond. output size:  (num_atoms, hidden_size)    (num_bonds, hidden_size)
                                                         (num_bonds, hidden_size)    (num_atoms, hidden_size)
        :param bias: enable bias term in all linear layers.
        :param cuda: run with cuda.
        :param res_connection: enables the skip-connection in MTBlock.
        """
        super(GTransEncoder, self).__init__()
        # For the compatibility issue.
        if atom_emb_output is False:
            atom_emb_output = None
        if atom_emb_output is True:
            atom_emb_output = 'atom'
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.cuda = cuda
        self.bias = bias
        self.res_connection = res_connection
        self.edge_blocks = nn.ModuleList()
        self.node_blocks = nn.ModuleList()
        edge_input_dim = edge_fdim
        node_input_dim = node_fdim
        edge_input_dim_i = edge_input_dim
        node_input_dim_i = node_input_dim
        for i in range(num_mt_block):
            if i != 0:
                edge_input_dim_i = self.hidden_size
                node_input_dim_i = self.hidden_size
            self.edge_blocks.append(MTBlock(args=args,
                                            num_attn_head=num_attn_head,
                                            input_dim=edge_input_dim_i,
                                            hidden_size=self.hidden_size,
                                            activation=activation,
                                            dropout=dropout,
                                            bias=self.bias,
                                            atom_messages=False,
                                            cuda=cuda))
            self.node_blocks.append(MTBlock(args=args,
                                            num_attn_head=num_attn_head,
                                            input_dim=node_input_dim_i,
                                            hidden_size=self.hidden_size,
                                            activation=activation,
                                            dropout=dropout,
                                            bias=self.bias,
                                            atom_messages=True,
                                            cuda=cuda))
        self.atom_emb_output = atom_emb_output
        self.ffn_atom_from_atom = PositionwiseFeedForward(self.hidden_size + node_fdim,
                                                          self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_out=self.hidden_size)
        self.ffn_atom_from_bond = PositionwiseFeedForward(self.hidden_size + node_fdim,
                                                          self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_out=self.hidden_size)
        self.ffn_bond_from_atom = PositionwiseFeedForward(self.hidden_size + edge_fdim,
                                                          self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_out=self.hidden_size)
        self.ffn_bond_from_bond = PositionwiseFeedForward(self.hidden_size + edge_fdim,
                                                          self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_out=self.hidden_size)
        self.atom_from_atom_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
        self.atom_from_bond_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
        self.bond_from_atom_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
        self.bond_from_bond_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
        self.dropout_layer = nn.Dropout(p=args.dropout)

    def pointwise_feed_forward_to_atom_embedding(self, emb_output, atom_fea, index, ffn_layer):
        """
        The point-wise feed forward and long-range residual connection for atom view.
        aggregate to atom.
        :param emb_output: the output embedding from the previous multi-head attentions.
        :param atom_fea: the atom/node feature embedding.
        :param index: the index of neighborhood relations.
        :param ffn_layer: the feed forward layer
        :return:
        """
        aggr_output = select_neighbor_and_aggregate(emb_output, index)
        aggr_outputx = torch.cat([atom_fea, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def pointwise_feed_forward_to_bond_embedding(self, emb_output, bond_fea, a2nei, b2revb, ffn_layer):
        """
        The point-wise feed forward and long-range residual connection for bond view.
        aggregate to bond.
        :param emb_output: the output embedding from the previous multi-head attentions.
        :param bond_fea: the bond/edge feature embedding.
        :param index: the index of neighborhood relations.
        :param ffn_layer: the feed forward layer
        :return:
        """
        aggr_output = select_neighbor_and_aggregate(emb_output, a2nei)
        # remove rev bond / atom --- need for bond view
        aggr_output = self.remove_rev_bond_message(emb_output, aggr_output, b2revb)
        aggr_outputx = torch.cat([bond_fea, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    @staticmethod
    def remove_rev_bond_message(orginal_message, aggr_message, b2revb):
        """
        :param orginal_message:
        :param aggr_message:
        :param b2revb:
        :return:
        """
        rev_message = orginal_message[b2revb]
        return aggr_message - rev_message

    def atom_bond_transform(self,
                            to_atom=True,  # False: to bond
                            atomwise_input=None,
                            bondwise_input=None,
                            original_f_atoms=None,
                            original_f_bonds=None,
                            a2a=None,
                            a2b=None,
                            b2a=None,
                            b2revb=None
                            ):
        """
        Transfer the output of atom/bond multi-head attention to the final atom/bond output.
        :param to_atom: if true, the output is atom emebedding, otherwise, the output is bond embedding.
        :param atomwise_input: the input embedding of atom/node.
        :param bondwise_input: the input embedding of bond/edge.
        :param original_f_atoms: the initial atom features.
        :param original_f_bonds: the initial bond features.
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :param a2b: mapping from atom index to incoming bond indices.
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :return:
        """
        if to_atom:
            # atom input to atom output
            atomwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(atomwise_input, original_f_atoms, a2a,
                                                                              self.ffn_atom_from_atom)
            atom_in_atom_out = self.atom_from_atom_sublayer(None, atomwise_input)
            # bond to atom
            bondwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(bondwise_input, original_f_atoms, a2b,
                                                                              self.ffn_atom_from_bond)
            bond_in_atom_out = self.atom_from_bond_sublayer(None, bondwise_input)
            return atom_in_atom_out, bond_in_atom_out
        else:  # to bond embeddings
            # atom input to bond output
            atom_list_for_bond = torch.cat([b2a.unsqueeze(dim=1), a2a[b2a]], dim=1)
            atomwise_input, _ = self.pointwise_feed_forward_to_bond_embedding(atomwise_input, original_f_bonds,
                                                                              atom_list_for_bond,
                                                                              b2a[b2revb], self.ffn_bond_from_atom)
            atom_in_bond_out = self.bond_from_atom_sublayer(None, atomwise_input)
            # bond input to bond output
            bond_list_for_bond = a2b[b2a]
            bondwise_input, _ = self.pointwise_feed_forward_to_bond_embedding(bondwise_input, original_f_bonds,
                                                                              bond_list_for_bond,
                                                                              b2revb, self.ffn_bond_from_bond)
            bond_in_bond_out = self.bond_from_bond_sublayer(None, bondwise_input)
            return atom_in_bond_out, bond_in_bond_out

    def forward(self, batch, features_batch = None):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
        if self.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()
            a2a = a2a.cuda()
        node_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        edge_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        # opt pointwise_feed_forward
        original_f_atoms, original_f_bonds = f_atoms, f_bonds
        # Note: features_batch is not used here.
        for nb in self.node_blocks:  # atom messages. Multi-headed attention
            node_batch, features_batch = nb(node_batch, features_batch)
        for eb in self.edge_blocks:  # bond messages. Multi-headed attention
            edge_batch, features_batch = eb(edge_batch, features_batch)
        atom_output, _, _, _, _, _, _, _ = node_batch  # atom hidden states
        _, bond_output, _, _, _, _, _, _ = edge_batch  # bond hidden states
        if self.atom_emb_output is None:
            # output the embedding from multi-head attention directly.
            return atom_output, bond_output
        if self.atom_emb_output == 'atom':
            return self.atom_bond_transform(to_atom=True,  # False: to bond
                                            atomwise_input=atom_output,
                                            bondwise_input=bond_output,
                                            original_f_atoms=original_f_atoms,
                                            original_f_bonds=original_f_bonds,
                                            a2a=a2a,
                                            a2b=a2b,
                                            b2a=b2a,
                                            b2revb=b2revb)
        elif self.atom_emb_output == 'bond':
            return self.atom_bond_transform(to_atom=False,  # False: to bond
                                            atomwise_input=atom_output,
                                            bondwise_input=bond_output,
                                            original_f_atoms=original_f_atoms,
                                            original_f_bonds=original_f_bonds,
                                            a2a=a2a,
                                            a2b=a2b,
                                            b2a=b2a,
                                            b2revb=b2revb)
        else:  # 'both'
            atom_embeddings = self.atom_bond_transform(to_atom=True,  # False: to bond
                                                       atomwise_input=atom_output,
                                                       bondwise_input=bond_output,
                                                       original_f_atoms=original_f_atoms,
                                                       original_f_bonds=original_f_bonds,
                                                       a2a=a2a,
                                                       a2b=a2b,
                                                       b2a=b2a,
                                                       b2revb=b2revb)
            bond_embeddings = self.atom_bond_transform(to_atom=False,  # False: to bond
                                                       atomwise_input=atom_output,
                                                       bondwise_input=bond_output,
                                                       original_f_atoms=original_f_atoms,
                                                       original_f_bonds=original_f_bonds,
                                                       a2a=a2a,
                                                       a2b=a2b,
                                                       b2a=b2a,
                                                       b2revb=b2revb)
            # Notice: need to be consistent with output format of DualMPNN encoder
            return ((atom_embeddings[0], bond_embeddings[0]),
                    (atom_embeddings[1], bond_embeddings[1]))

def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.
    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "Linear":
        return lambda x: x
    else:
        raise ValueError(f'Activation "{activation}" not supported.')

def index_select_nd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.
    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    return target

def select_neighbor_and_aggregate(feature, index):
    """
    The basic operation in message passing.
    Caution: the index_selec_ND would cause the reproducibility issue when performing the training on CUDA.
    See: https://pytorch.org/docs/stable/notes/randomness.html
    :param feature: the candidate feature for aggregate. (n_nodes, hidden)
    :param index: the selected index (neighbor indexes).
    :return:
    """
    neighbor = index_select_nd(feature, index)
    return neighbor.sum(dim=1)

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, activation="PReLU", dropout=0.1, d_out=None):
        """Initialization.
        :param d_model: the input dimension.
        :param d_ff: the hidden dimension.
        :param activation: the activation function.
        :param dropout: the dropout rate.
        :param d_out: the output dimension, the default value is equal to d_model.
        """
        super(PositionwiseFeedForward, self).__init__()
        if d_out is None:
            d_out = d_model
        # By default, bias is on.
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.act_func = get_activation_function(activation)
    def forward(self, x):
        """
        The forward function
        :param x: input tensor.
        :return:
        """
        return self.W_2(self.dropout(self.act_func(self.W_1(x))))

class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""
    def __init__(self, args: Namespace,
                 atom_messages: bool,
                 init_message_dim: int,
                 attached_fea_fdim: int,
                 hidden_size: int,
                 bias: bool,
                 depth: int,
                 dropout: float,
                 undirected: bool,
                 dense: bool,
                 aggregate_to_atom: bool,
                 attach_fea: bool,
                 input_layer="fc",
                 dynamic_depth='none'
                 ):
        """
        Initializes the MPNEncoder.
        :param args: the arguments.
        :param atom_messages: enables atom_messages or not.
        :param init_message_dim:  the initial input message dimension.
        :param attached_fea_fdim:  the attached feature dimension.
        :param hidden_size: the output message dimension during message passing.
        :param bias: the bias in the message passing.
        :param depth: the message passing depth.
        :param dropout: the dropout rate.
        :param undirected: the message passing is undirected or not.
        :param dense: enables the dense connections.
        :param attach_fea: enables the feature attachment during the message passing process.
        :param dynamic_depth: enables the dynamic depth. Possible choices: "none", "uniform" and "truncnorm"
        """
        super(MPNEncoder, self).__init__()
        self.init_message_dim = init_message_dim
        self.attached_fea_fdim = attached_fea_fdim
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.input_layer = input_layer
        self.layers_per_message = 1
        self.undirected = undirected
        self.atom_messages = atom_messages
        self.dense = dense
        self.aggreate_to_atom = aggregate_to_atom
        self.attached_fea = attach_fea
        self.dynamic_depth = dynamic_depth
        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # Activation
        self.act_func = get_activation_function(args.activation)
        # Input
        if self.input_layer == "fc":
            input_dim = self.init_message_dim
            self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        if self.attached_fea:
            w_h_input_size = self.hidden_size + self.attached_fea_fdim
        else:
            w_h_input_size = self.hidden_size
        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
        self.dynamic_depth = args.dynamic_depth

    def forward(self,
                init_messages,
                init_attached_features,
                a2nei,
                a2attached,
                b2a=None,
                b2revb=None,
                adjs=None
                ) -> torch.FloatTensor:
        """
        The forward function.
        :param init_messages:  initial massages, can be atom features or bond features.
        :param init_attached_features: initial attached_features.
        :param a2nei: the relation of item to its neighbors. For the atom message passing, a2nei = a2a. For bond
        messages a2nei = a2b
        :param a2attached: the relation of item to the attached features during message passing. For the atom message
        passing, a2attached = a2b. For the bond message passing a2attached = a2a
        :param b2a: remove the reversed bond in bond message passing
        :param b2revb: remove the revered atom in bond message passing
        :return: if aggreate_to_atom or self.atom_messages, return num_atoms x hidden.
        Otherwise, return num_bonds x hidden
        """
        # Input
        if self.input_layer == 'fc':
            input = self.W_i(init_messages)  # num_bonds x hidden_size # f_bond
            message = self.act_func(input)  # num_bonds x hidden_size
        elif self.input_layer == 'none':
            input = init_messages
            message = input
        attached_fea = init_attached_features  # f_atom / f_bond
        # dynamic depth
        # uniform sampling from depth - 1 to depth + 1
        # only works in training.
        if self.training and self.dynamic_depth != "none":
            if self.dynamic_depth == "uniform":
                # uniform sampling
                ndepth = numpy.random.randint(self.depth - 3, self.depth + 3)
            else:
                # truncnorm
                mu = self.depth
                sigma = 1
                lower = mu - 3 * sigma
                upper = mu + 3 * sigma
                X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                ndepth = int(X.rvs(1))
        else:
            ndepth = self.depth
        # Message passing
        for _ in range(ndepth - 1):
            if self.undirected:
                # two directions should be the same
                message = (message + message[b2revb]) / 2
            nei_message = select_neighbor_and_aggregate(message, a2nei)
            a_message = nei_message
            if self.attached_fea:
                attached_nei_fea = select_neighbor_and_aggregate(attached_fea, a2attached)
                a_message = torch.cat((nei_message, attached_nei_fea), dim=1)
            if not self.atom_messages:
                rev_message = message[b2revb]
                if self.attached_fea:
                    atom_rev_message = attached_fea[b2a[b2revb]]
                    rev_message = torch.cat((rev_message, atom_rev_message), dim=1)
                # Except reverse bond its-self(w) ! \sum_{k\in N(u) \ w}
                message = a_message[b2a] - rev_message  # num_bonds x hidden
            else:
                message = a_message
            message = self.W_h(message)
            if not self.dense:
                message = self.act_func(message)  # num_bonds x hidden_size
            else:
                message = self.act_func(input + message)
            message = self.dropout_layer(message)  # num_bonds x hidden
        output = message
        return output  # num_atoms x hidden
