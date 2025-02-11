"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import tempfile
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from fairchem.core.components.runner import Runner

from submitit import AutoExecutor
from submitit.helpers import Checkpointable, DelayedSubmission

from fairchem.core.common import distutils
from fairchem.core.common.utils import get_timestamp_uid, setup_env_vars

# this effects the cli only since the actual job will be run in subprocesses or remoe
logging.basicConfig(level=logging.INFO)


ALLOWED_TOP_LEVEL_KEYS = {"job", "runner"}


class SchedulerType(str, Enum):
    LOCAL = "local"
    SLURM = "slurm"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class SlurmConfig:
    mem_gb: int = 80
    timeout_hr: int = 168
    cpus_per_task: int = 8
    partition: Optional[str] = None  # noqa: UP007 python 3.9 requires Optional still
    qos: Optional[str] = None  # noqa: UP007 python 3.9 requires Optional still
    account: Optional[str] = None  # noqa: UP007 python 3.9 requires Optional still


@dataclass
class SchedulerConfig:
    mode: SchedulerType = SchedulerType.LOCAL
    ranks_per_node: int = 1
    num_nodes: int = 1
    num_jobs: int = 1
    slurm: SlurmConfig = field(default_factory=lambda: SlurmConfig)


@dataclass
class JobConfig:
    run_name: str = field(
        default_factory=lambda: get_timestamp_uid() + uuid.uuid4().hex.upper()[0:4]
    )
    timestamp_id: str = field(default_factory=lambda: get_timestamp_uid())
    run_dir: str = field(default_factory=lambda: tempfile.TemporaryDirectory().name)
    device_type: DeviceType = DeviceType.CUDA
    debug: bool = False
    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig)
    log_dir_name: str = "logs"
    checkpoint_dir_name: str = "checkpoint"
    config_file_name: str = "canonical_config.yaml"
    logger: Optional[dict] = None  # noqa: UP007 python 3.9 requires Optional still
    seed: int = 0
    deterministic: bool = False

    @property
    def log_dir(self) -> str:
        return os.path.join(self.run_dir, self.timestamp_id, self.log_dir_name)

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.run_dir, self.timestamp_id, self.checkpoint_dir_name)

    @property
    def config_path(self) -> str:
        return os.path.join(self.run_dir, self.timestamp_id, self.config_file_name)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_deterministic_mode() -> None:
    # this is required for full cuda deterministic mode
    logging.info("Setting deterministic mode!")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class Submitit(Checkpointable):
    def __call__(self, dict_config: DictConfig, **run_kwargs) -> None:
        self.config = dict_config
        self.job_config: JobConfig = OmegaConf.to_object(dict_config.job)
        setup_env_vars()
        distutils.setup(map_job_config_to_dist_config(self.job_config))
        self._init_logger()
        _set_seeds(self.job_config.seed)
        if self.job_config.deterministic:
            _set_deterministic_mode()

        runner: Runner = hydra.utils.instantiate(dict_config.runner)
        runner.job_config = self.job_config
        runner.load_state()
        runner.run(**run_kwargs)
        distutils.cleanup()

    def _init_logger(self) -> None:
        if (
            self.job_config.logger
            and distutils.is_master()
            and not self.job_config.debug
        ):
            # get a partial function from the config and instantiate wandb with it
            # currently this assume we use a wandb logger
            logger_initializer = hydra.utils.instantiate(self.job_config.logger)
            simple_config = OmegaConf.to_container(
                self.config, resolve=True, throw_on_missing=True
            )
            logger_initializer(
                config=simple_config,
                run_id=self.job_config.timestamp_id,
                run_name=self.job_config.run_name,
                log_dir=self.job_config.log_dir,
            )

    def checkpoint(self, *args, **kwargs) -> DelayedSubmission:
        # TODO: this is yet to be tested properly
        logging.info("Submitit checkpointing callback is triggered")
        new_runner = Submitit()
        self.runner.save_state()
        logging.info("Submitit checkpointing callback is completed")
        return DelayedSubmission(new_runner, self.config, self.cli_args)


def map_job_config_to_dist_config(job_cfg: JobConfig) -> dict:
    scheduler_config = job_cfg.scheduler
    return {
        "world_size": scheduler_config.num_nodes * scheduler_config.ranks_per_node,
        "distributed_backend": (
            "gloo" if job_cfg.device_type == DeviceType.CPU else "nccl"
        ),
        "submit": scheduler_config.mode == SchedulerType.SLURM,
        "summit": None,
        "cpu": job_cfg.device_type == DeviceType.CPU,
        "use_cuda_visibile_devices": True,
    }


def get_canonical_config(config: DictConfig) -> DictConfig:
    # check that each key other than the allowed top level keys are used in config
    # find all top level keys are not in the allowed set
    all_keys = set(config.keys()).difference(ALLOWED_TOP_LEVEL_KEYS)
    used_keys = set()
    for key in all_keys:
        # make a copy of all keys except the key in question
        copy_cfg = OmegaConf.create({k: v for k, v in config.items() if k != key})
        try:
            OmegaConf.resolve(copy_cfg)
        except InterpolationKeyError:
            # if this error is thrown, this means the key was actually required
            used_keys.add(key)

    unused_keys = all_keys.difference(used_keys)
    if unused_keys != set():
        raise ValueError(
            f"Found unused keys in the config: {unused_keys}, please remove them!, only keys other than {ALLOWED_TOP_LEVEL_KEYS} or ones that are used as variables are allowed."
        )

    # resolve the config to fully replace the variables and delete all top level keys except for the ALLOWED_TOP_LEVEL_KEYS
    OmegaConf.resolve(config)
    return OmegaConf.create(
        {k: v for k, v in config.items() if k in ALLOWED_TOP_LEVEL_KEYS}
    )


def get_hydra_config_from_yaml(
    config_yml: str, overrides_args: list[str]
) -> DictConfig:
    # Load the configuration from the file
    os.environ["HYDRA_FULL_ERROR"] = "1"
    config_directory = os.path.dirname(os.path.abspath(config_yml))
    config_name = os.path.basename(config_yml)
    hydra.initialize_config_dir(config_directory, version_base="1.1")
    return hydra.compose(config_name=config_name, overrides=overrides_args)


def main(
    args: argparse.Namespace | None = None, override_args: list[str] | None = None
):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", type=str, required=True)
        args, override_args = parser.parse_known_args()

    cfg = get_hydra_config_from_yaml(args.config, override_args)
    # merge default structured config with initialized job object
    cfg = OmegaConf.merge({"job": OmegaConf.structured(JobConfig)}, cfg)
    # canonicalize config (remove top level keys that just used replacing variables)
    cfg = get_canonical_config(cfg)
    job_obj = OmegaConf.to_object(cfg.job)
    log_dir = job_obj.log_dir
    os.makedirs(job_obj.run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    OmegaConf.save(cfg, job_obj.config_path)
    logging.info(f"saved canonical config to {job_obj.config_path}")

    scheduler_cfg = job_obj.scheduler
    logging.info(f"Running fairchemv2 cli with {cfg}")
    if scheduler_cfg.mode == SchedulerType.SLURM:  # Run on cluster
        executor = AutoExecutor(folder=log_dir, slurm_max_num_timeout=3)
        executor.update_parameters(
            name=job_obj.run_name,
            mem_gb=scheduler_cfg.slurm.mem_gb,
            timeout_min=scheduler_cfg.slurm.timeout_hr * 60,
            slurm_partition=scheduler_cfg.slurm.partition,
            gpus_per_node=scheduler_cfg.ranks_per_node,
            cpus_per_task=scheduler_cfg.slurm.cpus_per_task,
            tasks_per_node=scheduler_cfg.ranks_per_node,
            nodes=scheduler_cfg.num_nodes,
            slurm_qos=scheduler_cfg.slurm.qos,
            slurm_account=scheduler_cfg.slurm.account,
        )
        if scheduler_cfg.num_jobs == 1:
            job = executor.submit(Submitit(), cfg)
            logging.info(
                f"Submitted job id: {job_obj.timestamp_id}, slurm id: {job.job_id}, logs: {job_obj.log_dir}"
            )
        elif scheduler_cfg.num_jobs > 1:
            executor.update_parameters(slurm_array_parallelism=scheduler_cfg.num_jobs)

            jobs = []
            with executor.batch():
                for job_number in range(scheduler_cfg.num_jobs):
                    job = executor.submit(
                        Submitit(),
                        cfg,
                        job_number=job_number,
                        num_jobs=scheduler_cfg.num_jobs,
                    )
                    jobs.append(job)
            logging.info(f"Submitted {len(jobs)} jobs: {jobs[0].job_id.split('_')[0]}")
    else:
        from torch.distributed.launcher.api import LaunchConfig, elastic_launch

        if scheduler_cfg.ranks_per_node > 1:
            logging.info(
                f"Running in local mode with {scheduler_cfg.ranks_per_node} ranks"
            )
            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=scheduler_cfg.ranks_per_node,
                rdzv_backend="c10d",
                max_restarts=0,
            )
            elastic_launch(launch_config, Submitit())(cfg)
        else:
            logging.info("Running in local mode without elastic launch")
            distutils.setup_env_local()
            Submitit()(cfg)
