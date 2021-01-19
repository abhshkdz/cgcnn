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


class RelaxationOptimizer:
    def __init__(
        self,
        atoms: Atoms,
        model,
        maxstep=0.01,
        damping=0.25,
        device="cuda:0",
        traj_dir: Path = None,
        traj_names=None,
        verbose=True,
    ):
        self.atoms = atoms
        self.model = model
        self.maxstep = maxstep
        self.damping = damping
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

    def determine_step(self, dr):
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

    def setup(self):
        raise NotImplementedError()

    def step(self, iteration, f0, r0):
        raise NotImplementedError()

    def run(self, fmax, steps):
        self.setup()
        r0 = f0 = e0 = None
        trajectories = None

        if self.traj_dir:
            self.traj_dir.mkdir(exist_ok=True, parents=True)
            trajectories = [
                ase.io.Trajectory(self.traj_dir / f"{name}.traj", mode="a")
                for name in self.traj_names
            ]

        iteration = 0
        while iteration < steps and not self.converged(fmax, iteration, f0):
            r0, f0, e0 = self.step(iteration, f0, r0)
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


class Adam(RelaxationOptimizer):
    def __init__(
        self,
        atoms: Atoms,
        model,
        maxstep=0.01,
        damping=0.25,
        device="cuda:0",
        traj_dir: Path = None,
        traj_names=None,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        verbose=True
    ):
        super().__init__(atoms, model, maxstep, damping, device, traj_dir, traj_names, verbose)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.setup()

    def setup(self):
        self.m = 0.
        self.v = 0.

    def step(self, iteration, f0, r0):
        e, f = self.get_forces()
        f = f.to(self.device, dtype=torch.float64)
        r = self.atoms.pos.to(self.device, dtype=torch.float64)

        dr = torch.clone(f)

        self.m = self.beta1 * self.m + (1-self.beta1) * dr
        mt = self.m / (1-self.beta1**(iteration+1))
        self.v = self.beta2 * self.v + (1-self.beta2) * (dr**2)
        vt = self.v / (1-self.beta2**(iteration+1))
        dr = mt / (torch.sqrt(vt) + self.eps)

        dr.mul_(self.lr)
        dr = self.determine_step(dr)
        self.set_positions(dr)
        return r, f, e


class GradientDescent(RelaxationOptimizer):
    def __init__(
        self,
        atoms: Atoms,
        model,
        maxstep=0.01,
        damping=0.25,
        device="cuda:0",
        traj_dir: Path = None,
        traj_names=None,
        lr=0.001,
        mu=0.,
        nesterov=False,
        verbose=True
    ):
        super().__init__(atoms, model, maxstep, damping, device, traj_dir, traj_names, verbose)
        self.lr = lr
        self.mu = mu
        self.nesterov = nesterov
        self.setup()

    def setup(self):
        self.momentum = None

    def step(self, iteration, f0, r0):
        e, f = self.get_forces()
        f = f.to(self.device, dtype=torch.float64)
        r = self.atoms.pos.to(self.device, dtype=torch.float64)

        dr = torch.clone(f)
        if self.mu != 0:
            if self.momentum is None:
                self.momentum = torch.clone(dr).detach()
            else:
                self.momentum.mul_(self.mu).add_(dr)

            if self.nesterov:
                dr.add_(self.momentum, alpha=self.mu)
            else:
                dr = self.momentum
        dr.mul_(self.lr)
        dr = self.determine_step(dr)
        self.set_positions(dr)
        return r, f, e


class LBFGS(RelaxationOptimizer):
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
        verbose=True
    ):
        super().__init__(atoms, model, maxstep, damping, device, traj_dir, traj_names, verbose)
        self.memory = memory
        self.alpha = alpha
        self.force_consistent = force_consistent
        self.setup()

    def setup(self):
        self.s = deque(maxlen=self.memory)
        self.y = deque(maxlen=self.memory)
        self.rho = deque(maxlen=self.memory)
        self.H0 = 1.0 / self.alpha

    def step(self, iteration, f0, r0):
        e, f = self.get_forces()
        f = f.to(self.device, dtype=torch.float64)
        r = self.atoms.pos.to(self.device, dtype=torch.float64)

        # Update s, y and rho
        if iteration > 0:
            s0 = (r - r0).flatten()
            y0 = -(f - f0).flatten()
            self.s.append(s0)
            self.y.append(y0)
            self.rho.append(1.0 / torch.dot(y0, s0))

        loopmax = min(self.memory, iteration)
        alpha = f.new_empty(loopmax)
        q = -f.flatten()

        for i in range(loopmax - 1, -1, -1):
            alpha[i] = self.rho[i] * torch.dot(self.s[i], q)
            q -= alpha[i] * self.y[i]
        z = self.H0 * q
        for i in range(loopmax):
            beta = self.rho[i] * torch.dot(self.y[i], z)
            z += self.s[i] * (alpha[i] - beta)
        p = -z.reshape((-1, 3))  # descent direction
        dr = self.determine_step(p)
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
