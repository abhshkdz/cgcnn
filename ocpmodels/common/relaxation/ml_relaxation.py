"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

import torch

from ocpmodels.common.meter import mae, mae_ratio, mean_l2_distance
from ocpmodels.common.registry import registry

from .optimizers.lbfgs_torch import LBFGS, GradientDescent, Adam, TorchCalc


def ml_relax(
    batch,
    model,
    steps,
    fmax,
    relax_opt,
    device="cuda:0",
    transform=None,
    verbose=True,
):
    """
    Runs ML-based relaxations.
    Args:
        batch: object
        model: object
        steps: int
            Max number of steps in the structure relaxation.
        fmax: float
            Structure relaxation terminates when the max force
            of the system is no bigger than fmax.
        relax_opt: str
            Optimizer and corresponding parameters to be used for structure relaxations.
    """
    batch = batch[0]
    ids = batch.sid
    calc = TorchCalc(model, transform)

    # Run ML-based relaxation
    traj_dir = relax_opt.get("traj_dir", None)
    if traj_dir is not None:
        traj_dir = Path(traj_dir)
    if relax_opt["optimizer"] == "lbfgs":
        optimizer = LBFGS(
            batch,
            calc,
            maxstep=relax_opt.get("maxstep", 0.04),
            memory=relax_opt["memory"],
            damping=relax_opt.get("damping", 1.0),
            alpha=relax_opt.get("alpha", 70.0),
            device=device,
            traj_dir=traj_dir,
            traj_names=ids,
            verbose=False
        )
    elif relax_opt["optimizer"] == "gd":
        optimizer = GradientDescent(
            batch,
            calc,
            maxstep=relax_opt.get("maxstep", 0.04),
            damping=relax_opt.get("damping", 1.0),
            device=device,
            traj_dir=traj_dir,
            traj_names=ids,
            lr=relax_opt.get("lr", 0.001),
            mu=0,
            nesterov=relax_opt.get("nesterov", False),
            verbose=False
        )
    elif relax_opt["optimizer"] == "gd_momentum":
        optimizer = GradientDescent(
            batch,
            calc,
            maxstep=relax_opt.get("maxstep", 0.04),
            damping=relax_opt.get("damping", 1.0),
            device=device,
            traj_dir=traj_dir,
            traj_names=ids,
            lr=relax_opt.get("lr", 0.001),
            mu=relax_opt.get("mu", 0.99),
            nesterov=relax_opt.get("nesterov", False),
            verbose=False
        )
    elif relax_opt["optimizer"] == "gd_nesterov_momentum":
        optimizer = GradientDescent(
            batch,
            calc,
            maxstep=relax_opt.get("maxstep", 0.04),
            damping=relax_opt.get("damping", 1.0),
            device=device,
            traj_dir=traj_dir,
            traj_names=ids,
            lr=relax_opt.get("lr", 0.001),
            mu=relax_opt.get("mu", 0.99),
            nesterov=relax_opt.get("nesterov", True),
            verbose=False
        )
    elif relax_opt["optimizer"] == "adam":
        optimizer = Adam(
            batch,
            calc,
            maxstep=relax_opt.get("maxstep", 0.04),
            damping=relax_opt.get("damping", 1.0),
            device=device,
            traj_dir=traj_dir,
            traj_names=ids,
            lr=relax_opt.get("lr", 0.001),
            beta1=relax_opt.get("mu", 0.9),
            beta2=relax_opt.get("mu", 0.999),
            eps=relax_opt.get("eps", 1e-8),
            verbose=False
        )
    else:
        raise ValueError

    relaxed_batch = optimizer.run(fmax=fmax, steps=steps)

    return relaxed_batch
