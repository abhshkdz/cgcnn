"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
from typing import Dict, Union

from ocpmodels.common.utils import change_mat

"""
An evaluation module for use with the OCP dataset and suite of tasks. It should
be possible to import this independently of the rest of the codebase, e.g:

```
from ocpmodels.modules import Evaluator

evaluator = Evaluator(task="is2re")
perf = evaluator.eval(prediction, target)
```

task: "s2ef", "is2rs", "is2re".

We specify a default set of metrics for each task, but should be easy to extend
to add more metrics. `evaluator.eval` takes as input two dictionaries, one for
predictions and another for targets to check against. It returns a dictionary
with the relevant metrics computed.
"""


class Evaluator:
    task_metrics = {
        "s2ef": {
            "energy": {"metrics": ["energy_mae"]},
            "forces": {
                "metrics": [
                    "forcesx_mae",
                    "forcesy_mae",
                    "forcesz_mae",
                    "forces_mae",
                    "forces_cos",
                    "forces_magnitude",
                    "energy_force_within_threshold",
                ]
            },
        },
        "is2rs": {
            "positions": {
                "metrics": [
                    "average_distance_within_threshold",
                    "positions_mae",
                    "positions_mse",
                ]
            }
        },
        "is2re": {
            "metrics": [
                "energy_mae",
                "energy_mse",
                "energy_within_threshold",
            ]
        },
    }

    task_primary_metric = {
        "s2ef": "energy_force_within_threshold",
        "is2rs": "average_distance_within_threshold",
        "is2re": "energy_mae",
    }

    def __init__(self, task: str = None, eval_metrics: str = None) -> None:
        self.task = task
        self.target_metrics = self.task_metrics.get(task, eval_metrics)

    def eval(self, prediction, target, prev_metrics={}):

        # TODO: arbitrary type check
        # for metric in self.metric_fns:
        # assert attr in prediction
        # assert attr in target
        # assert prediction[attr].shape == target[attr].shape

        metrics = prev_metrics

        for target_property in self.target_metrics:
            for fn in self.target_metrics[target_property]["metrics"]:
                metric_name = (
                    f"{target_property}_{fn}"
                    if target_property not in fn
                    else fn
                )
                res = eval(fn)(prediction, target, target_property)
                metrics = self.update(metric_name, res, metrics)

        return metrics

    def update(self, key, stat, metrics):
        if key not in metrics:
            metrics[key] = {
                "metric": None,
                "total": 0,
                "numel": 0,
            }

        if isinstance(stat, dict):
            # If dictionary, we expect it to have `metric`, `total`, `numel`.
            metrics[key]["total"] += stat["total"]
            metrics[key]["numel"] += stat["numel"]
            metrics[key]["metric"] = (
                metrics[key]["total"] / metrics[key]["numel"]
            )
        elif isinstance(stat, float) or isinstance(stat, int):
            # If float or int, just add to the total and increment numel by 1.
            metrics[key]["total"] += stat
            metrics[key]["numel"] += 1
            metrics[key]["metric"] = (
                metrics[key]["total"] / metrics[key]["numel"]
            )
        elif torch.is_tensor(stat):
            raise NotImplementedError

        return metrics


def forcesx_mae(prediction, target, key=None):
    return mae(prediction["forces"][:, 0], target["forces"][:, 0])


def forcesx_mse(prediction, target, key=None):
    return mse(prediction["forces"][:, 0], target["forces"][:, 0])


def forcesy_mae(prediction, target, key=None):
    return mae(prediction["forces"][:, 1], target["forces"][:, 1])


def forcesy_mse(prediction, target, key=None):
    return mse(prediction["forces"][:, 1], target["forces"][:, 1])


def forcesz_mae(prediction, target, key=None):
    return mae(prediction["forces"][:, 2], target["forces"][:, 2])


def forcesz_mse(prediction, target, key=None):
    return mse(prediction["forces"][:, 2], target["forces"][:, 2])


<<<<<<< HEAD
def forces_mae(prediction, target):
    return mae(prediction["forces"], target["forces"])


def forces_mse(prediction, target):
    return mse(prediction["forces"], target["forces"])


def forces_cos(prediction, target):
    return cosine_similarity(prediction["forces"], target["forces"])


def forces_magnitude(prediction, target):
    return magnitude_error(prediction["forces"], target["forces"], p=2)


def positions_mae(prediction, target):
    return mae(prediction["positions"], target["positions"])


def positions_mse(prediction, target):
    return mse(prediction["positions"], target["positions"])


def energy_force_within_threshold(
    prediction, target, key=None
) -> Dict[str, Union[float, int]]:
    # Note that this natoms should be the count of free atoms we evaluate over.
    assert target["natoms"].sum() == prediction["forces"].size(0)
    assert target["natoms"].size(0) == prediction["energy"].size(0)

    # compute absolute error on per-atom forces and energy per system.
    # then count the no. of systems where max force error is < 0.03 and max
    # energy error is < 0.02.
    f_thresh = 0.03
    e_thresh = 0.02

    success = 0
    total = int(target["natoms"].size(0))

    error_forces = torch.abs(target["forces"] - prediction["forces"])
    error_energy = torch.abs(target["energy"] - prediction["energy"])

    start_idx = 0
    for i, n in enumerate(target["natoms"]):
        if (
            error_energy[i] < e_thresh
            and error_forces[start_idx : start_idx + n].max() < f_thresh
        ):
            success += 1
        start_idx += n

    return {
        "metric": success / total,
        "total": success,
        "numel": total,
    }


def energy_within_threshold(
    prediction, target, key=None
) -> Dict[str, Union[float, int]]:
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.
    e_thresh = 0.02
    error_energy = torch.abs(target["energy"] - prediction["energy"])

    success = (error_energy < e_thresh).sum().item()
    total = target["energy"].size(0)

    return {
        "metric": success / total,
        "total": success,
        "numel": total,
    }


def average_distance_within_threshold(
    prediction, target, key=None
) -> Dict[str, Union[float, int]]:
    pred_pos = torch.split(
        prediction["positions"], prediction["natoms"].tolist()
    )
    target_pos = torch.split(target["positions"], target["natoms"].tolist())

    mean_distance = []
    for idx, ml_pos in enumerate(pred_pos):
        mean_distance.append(
            np.mean(
                np.linalg.norm(
                    min_diff(
                        ml_pos.detach().cpu().numpy(),
                        target_pos[idx].detach().cpu().numpy(),
                        target["cell"][idx].detach().cpu().numpy(),
                        target["pbc"].tolist(),
                    ),
                    axis=1,
                )
            )
        )

    success = 0
    intv = np.arange(0.01, 0.5, 0.001)
    for i in intv:
        success += sum(np.array(mean_distance) < i)

    total = len(mean_distance) * len(intv)

    return {"metric": success / total, "total": success, "numel": total}


def stress_mae(prediction, target, key=None):
    device = prediction["isotropic_stress"].device
    cg_decomp_mat = change_mat.to(device)

    zero_vectors = torch.zeros(
        (prediction["isotropic_stress"].shape[0], 3),
        device=device,
    )
    prediction_irreps = torch.concat(
        [
            prediction["isotropic_stress"].reshape(-1, 1),
            zero_vectors,
            prediction["anisotropic_stress"].reshape(-1, 5),
        ],
        dim=1,
    )
    prediction_stress = torch.einsum(
        "ba, cb->ca", cg_decomp_mat, prediction_irreps
    ).reshape(-1)

    target_stress = target["stress"]

    return mae(prediction_stress, target_stress)


def min_diff(pred_pos, dft_pos, cell, pbc):
    pos_diff = pred_pos - dft_pos
    fractional = np.linalg.solve(cell.T, pos_diff.T).T

    for i, periodic in enumerate(pbc):
        # Yes, we need to do it twice
        if periodic:
            fractional[:, i] %= 1.0
            fractional[:, i] %= 1.0

    fractional[fractional > 0.5] -= 1

    return np.matmul(fractional, cell)


def cosine_similarity(prediction: dict, target: dict, key=slice(None)):
    error = torch.cosine_similarity(prediction[key], target[key])
    return {
        "metric": torch.mean(error).item(),
        "total": torch.sum(error).item(),
        "numel": error.numel(),
    }


def mae(
    prediction: dict, target: dict, key=slice(None)
) -> Dict[str, Union[float, int]]:
    error = torch.abs(target[key] - prediction[key])
    return {
        "metric": torch.mean(error).item(),
        "total": torch.sum(error).item(),
        "numel": error.numel(),
    }


def mse(
    prediction: dict, target: dict, key=slice(None)
) -> Dict[str, Union[float, int]]:
    error = (target[key] - prediction[key]) ** 2
    return {
        "metric": torch.mean(error).item(),
        "total": torch.sum(error).item(),
        "numel": error.numel(),
    }


def magnitude_error(
    prediction: dict, target: dict, key=slice(None), p: int = 2
) -> Dict[str, Union[float, int]]:
    assert prediction.shape[1] > 1
    error = torch.abs(
        torch.norm(prediction[key], p=p, dim=-1)
        - torch.norm(target[key], p=p, dim=-1)
    )
    return {
        "metric": torch.mean(error).item(),
        "total": torch.sum(error).item(),
        "numel": error.numel(),
    }
