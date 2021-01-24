"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import pickle
from pathlib import Path

import ase.io
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.preprocessing import AtomsToGraphs


@registry.register_dataset("trajectory_ase")
class TrajectoryASEDataset(Dataset):
    r"""Dataset class to load from ASE relaxation trajetories.
    Useful for Structure to Energy & Force (S2EF) and potentially Initial State to
    Relaxed State (IS2RS) tasks.

    Args:
        config (dict): Dataset configuration
        lazy (bool, optional): Allow lazy loading of data.
            (default: :obj:`False`)
    """

    def __init__(self, config, lazy=False):
        super(TrajectoryASEDataset, self).__init__()
        self.config = config

        self.lazy = lazy
        src_path = Path(self.config["src"])

        self.a2g = AtomsToGraphs(
            max_neigh=50,
            radius=6,
            r_energy=True,
            r_forces=True,
            r_fixed=True,
            r_distances=False,
            r_edges=False,
        )

        # for example the input could be: "/checkpoint/electrocatalysis/relaxations/mapping/EF/train_ef.txt"
        # this stores a list of unique trajectory paths
        self.traj_paths = self.get_unique_traj_paths(src_path)

        self.reference_energy_map = pickle.load(
            open(
                "/checkpoint/electrocatalysis/relaxations/mapping/new_adslab_ref_energies_09_22_20.pkl",
                "rb",
            )
        )

        self.precomputed = []
        if not self.lazy:
            for i in range(len(self.traj_paths)):
                self.precomputed.append(self.get_info_per_system(i))

    def get_unique_traj_paths(self, fname):
        outputs = set()
        size = self.config.get("size", -1)
        with open(fname, "r") as f:
            for idx, line in enumerate(f):
                if idx == size:
                    break
                temp = line.rstrip().split(",")[0]
                outputs.add(temp)
        return list(outputs)

    def __len__(self):
        return len(self.traj_paths)

    def __getitem__(self, idx):
        if self.lazy:
            return self.get_info_per_system(idx)
        else:
            return self.precomputed[idx]

    def extract_system_from_traj_path(self, pathname):
        return pathname.split("/")[-1].split(".")[0]

    def get_info_per_system(self, idx):
        # read underlying ASE trajectory
        images = ase.io.read(self.traj_paths[idx], ":")

        data_object_list = []

        system_id = self.extract_system_from_traj_path(self.traj_paths[idx])
        reference_energy = self.reference_energy_map[system_id]
        relaxed_data_object = self.a2g.convert(images[-1])

        for img in images:  # each img is an ASE atoms object.
            cur_data_object = self.a2g.convert(
                img
            )  # convert returns a `torch_geometric.data.Data` object

            cur_data_object.sid = system_id
            cur_data_object.y_relaxed = (
                relaxed_data_object.y - reference_energy
            )
            cur_data_object.pos_relaxed = relaxed_data_object.pos
            cur_data_object.y -= reference_energy
            data_object_list.append(cur_data_object)

        d = {}
        d["sid"] = system_id
        d["data_objects"] = data_object_list

        return d
