import os

import numpy as np
import submitit

from ocpmodels.datasets import *
from ocpmodels.trainers import MDTrainer


def main_helper():
    task = {
        "dataset": "co_cu_md",
        "description": "Regressing to binding energies for an MD trajectory of CO on Cu",
        "labels": ["potential energy"],
        "metric": "mae",
        "type": "regression",
        "grad_input": "atomic forces",
        # whether to multiply / scale gradient wrt input
        "grad_input_mult": -1,
        # indexing which attributes in the input vector to compute gradients for.
        # data.x[:, grad_input_start_idx:grad_input_end_idx]
        "grad_input_start_idx": 92,
        "grad_input_end_idx": 95,
    }

    model = {
        "name": "cnn3d_local",
        "regress_forces": True,
        "max_num_elements": 3,
    }

    dataset = {
        "src": "data/data/2020_04_14_muhammed_md",
        "traj": "COCu_DFT_10ps.traj",
        "train_size": 1000,
        "val_size": 500,
        "test_size": 2000,
        "normalize_labels": True,
    }

    optimizer = {
        "batch_size": 16,
        "lr_gamma": 0.1,
        "lr_initial": 0.001,
        "lr_milestones": [8, 50],
        "max_epochs": 100,
        "warmup_epochs": 1,
        "warmup_factor": 0.2,
    }
    trainer = MDTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier="p_energy_with_positions_forces_1ktrain_seed2",
        print_every=5,
        is_debug=False,
        seed=2,
    )

    trainer.train()

    dataset_config = {
        "src": "data/data/2020_04_14_muhammed_md",
        "traj": "COCu_DFT_10ps.traj",
    }
    predictions = trainer.predict(dataset_config)
    np.save(
        os.path.join(trainer.config["cmd"]["results_dir"], "preds.npy"),
        predictions,
    )


if __name__ == "__main__":
    main_helper()

    # executor = submitit.AutoExecutor(folder="logs")
    # executor.update_parameters(timeout_min=1, slurm_partition="learnfair")
    # job = executor.submit(main_helper)
    # print(job.job_id)
