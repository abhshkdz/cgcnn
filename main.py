"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import os
import time
from pathlib import Path

import submitit

from ocpmodels.common import distutils
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_imports,
)
from ocpmodels.trainers import ForcesTrainer


class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None
        self.chkpt_path = None

    def __call__(self, config):
        self.config = copy.deepcopy(config)

        if args.distributed:
            distutils.setup(config)

        try:
            setup_imports()
            trainer = registry.get_trainer_class(
                config.get("trainer", "simple")
            )(
                task=config["task"],
                model=config["model"],
                dataset=config["dataset"],
                optimizer=config["optim"],
                identifier=config["identifier"],
                run_dir=config.get("run_dir", "./"),
                is_debug=config.get("is_debug", False),
                is_vis=config.get("is_vis", False),
                print_every=config.get("print_every", 10),
                seed=config.get("seed", 0),
                logger=config.get("logger", "tensorboard"),
                local_rank=config["local_rank"],
                amp=config.get("amp", False),
                cpu=config.get("cpu", False),
            )
            if config["checkpoint"] is not None:
                trainer.load_pretrained(config["checkpoint"])

            # save checkpoint path to runner state for slurm resubmissions
            self.chkpt_path = os.path.join(
                trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
            )

            start_time = time.time()

            if config["mode"] == "train":
                trainer.train()

            elif config["mode"] == "predict":
                assert (
                    trainer.test_loader is not None
                ), "Test dataset is required for making predictions"
                assert config["checkpoint"]
                results_file = "predictions"
                trainer.predict(
                    trainer.test_loader,
                    results_file=results_file,
                    disable_tqdm=False,
                )

            elif config["mode"] == "run-relaxations":
                assert isinstance(
                    trainer, ForcesTrainer
                ), "Relaxations are only possible for ForcesTrainer"
                assert (
                    trainer.relax_dataset is not None
                ), "Relax dataset is required for making predictions"
                assert config["checkpoint"]
                trainer.run_relaxations()

            distutils.synchronize()

            if distutils.is_master():
                print("Total time taken = ", time.time() - start_time)

        finally:
            if args.distributed:
                distutils.cleanup()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        if os.path.isfile(self.chkpt_path):
            self.config["checkpoint"] = self.chkpt_path
        return submitit.helpers.DelayedSubmission(new_runner, self.config)


if __name__ == "__main__":
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    if args.submit:  # Run on cluster
        if args.sweep_yml:  # Run grid search
            configs = create_grid(config, args.sweep_yml)
        else:
            configs = [config]

        print(f"Submitting {len(configs)} jobs")
        executor = submitit.AutoExecutor(
            folder=args.logdir / "%j", slurm_max_num_timeout=3
        )
        executor.update_parameters(
            name=args.identifier,
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(args.num_workers + 1),
            tasks_per_node=1,  # (args.num_gpus if args.distributed else 1), For DDP at inter-node-level, we only need to launch one task per node; intra-node computation will happen via pipeline parallelism
            nodes=args.num_nodes,
        )
        jobs = executor.map_array(Runner(), configs)
        print("Submitted jobs:", ", ".join([job.job_id for job in jobs]))
        log_file = save_experiment_log(args, jobs, configs)
        print(f"Experiment log saved to: {log_file}")

    else:  # Run locally
        Runner()(config)
