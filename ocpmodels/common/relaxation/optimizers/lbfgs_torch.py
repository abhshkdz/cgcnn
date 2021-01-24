"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import deque
from pathlib import Path

import ase
import torch
from ase import Atoms
from torch_scatter import scatter

from ocpmodels.common.relaxation.ase_utils import batch_to_atoms
from ocpmodels.common.utils import radius_graph_pbc


class LBFGS:
    def __init__(
        self,
        atoms: Atoms,
        model,
        maxstep=0.01,
        memory=100,
        damping=0.25,
        alpha=100.0,
        force_consistent=None,
        device="cuda:0",
        traj_dir: Path = None,
        traj_names=None,
        verbose=True,
    ):
        self.atoms = atoms
        self.model = model
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping
        self.alpha = alpha
        self.force_consistent = force_consistent
        self.device = device
        self.traj_dir = traj_dir
        self.traj_names = traj_names
        self.verbose = verbose
        assert not self.traj_dir or (
            traj_dir and len(traj_names)
        ), "Trajectory names should be specified to save trajectories"
        if self.verbose:
            print("Step   Fmax(eV/A)")

        self.model.update_graph(self.atoms)

    def get_forces(self, apply_constraint=True):
        energy, forces = self.model.get_forces(self.atoms, apply_constraint)
        return energy, forces

    def get_positions(self):
        return self.atoms.pos

    def set_positions(self, update):
        r = self.get_positions()
        self.atoms.pos = r + update.to(dtype=torch.float32)
        self.model.update_graph(self.atoms)

    def converged(self, force_threshold, iteration, forces):
        if forces is None:
            return False
        if self.verbose:
            print(
                iteration, torch.sqrt((forces ** 2).sum(axis=1).max()).item()
            )
        return (forces ** 2).sum(axis=1).max() < force_threshold ** 2

    def run(self, fmax, steps):
        s = deque(maxlen=self.memory)
        y = deque(maxlen=self.memory)
        rho = deque(maxlen=self.memory)
        r0 = f0 = e0 = None
        H0 = 1.0 / self.alpha

        trajectories = None
        if self.traj_dir:
            self.traj_dir.mkdir(exist_ok=True, parents=True)
            trajectories = [
                ase.io.Trajectory(self.traj_dir / f"{name}.traj", mode="a")
                for name in self.traj_names
            ]

        iteration = 0
        while iteration < steps and not self.converged(fmax, iteration, f0):
            r0, f0, e0 = self.step(iteration, r0, f0, H0, rho, s, y)
            iteration += 1
            if trajectories is not None:
                self.atoms.y, self.atoms.force = e0, f0
                atoms_objects = batch_to_atoms(self.atoms)
                for atm, traj in zip(atoms_objects, trajectories):
                    traj.write(atm)

        if trajectories is not None:
            for traj in trajectories:
                traj.close()

        self.atoms.y, self.atoms.force = self.get_forces(
            apply_constraint=False
        )
        return self.atoms

    def step(self, iteration, r0, f0, H0, rho, s, y):
        def determine_step(dr):
            steplengths = torch.norm(dr, dim=1)
            longest_steps = scatter(
                steplengths, self.atoms.batch, reduce="max"
            )
            longest_steps = torch.repeat_interleave(
                longest_steps, self.atoms.natoms
            )
            maxstep = longest_steps.new_tensor(self.maxstep)
            scale = (longest_steps + 1e-7).reciprocal() * torch.min(
                longest_steps, maxstep
            )
            dr *= scale.unsqueeze(1)
            return dr * self.damping

        e, f = self.get_forces()
        f = f.to(self.device, dtype=torch.float64)
        r = self.atoms.pos.to(self.device, dtype=torch.float64)

        # Update s, y and rho
        if iteration > 0:
            s0 = (r - r0).flatten()
            y0 = -(f - f0).flatten()
            s.append(s0)
            y.append(y0)
            rho.append(1.0 / torch.dot(y0, s0))

        loopmax = min(self.memory, iteration)
        alpha = f.new_empty(loopmax)
        q = -f.flatten()

        for i in range(loopmax - 1, -1, -1):
            alpha[i] = rho[i] * torch.dot(s[i], q)
            q -= alpha[i] * y[i]
        z = H0 * q
        for i in range(loopmax):
            beta = rho[i] * torch.dot(y[i], z)
            z += s[i] * (alpha[i] - beta)
        p = -z.reshape((-1, 3))  # descent direction
        dr = determine_step(p)
        if torch.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return
        self.set_positions(dr)
        return r, f, e


class TorchCalc:
    def __init__(self, model, transform=None):
        self.model = model
        self.transform = transform

    def get_forces(self, atoms, apply_constraint=True):
        predictions = self.model.predict(atoms, per_image=False)
        energy = predictions["energy"]
        forces = predictions["forces"]
        if apply_constraint:
            fixed_idx = torch.where(atoms.fixed == 1)[0]
            forces[fixed_idx] = 0
        return energy, forces

    def update_graph(self, atoms):
        edge_index, cell_offsets, num_neighbors = radius_graph_pbc(
            atoms, 6, 50, atoms.pos.device
        )
        atoms.edge_index = edge_index
        atoms.cell_offsets = cell_offsets
        atoms.neighbors = num_neighbors
        if self.transform is not None:
            atoms = self.transform(atoms)
        return atoms
