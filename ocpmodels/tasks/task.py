"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os

from ocpmodels.common.registry import registry
from ocpmodels.trainers.forces_trainer import ForcesTrainer


class BaseTask:
    def __init__(self, config):
        self.config = config

    def setup(self, trainer):
        self.trainer = trainer
        if self.config.get("checkpoint") is not None:
            print("\n🔵 Resuming:\n  • ", end="", flush=True)
            self.trainer.load_checkpoint(self.config["checkpoint"])
            print()

        # save checkpoint path to runner state for slurm resubmissions
        self.chkpt_path = os.path.join(
            self.trainer.config["checkpoint_dir"], "checkpoint.pt"
        )

    def run(self):
        raise NotImplementedError


@registry.register_task("train")
class TrainTask(BaseTask):
    def _process_error(self, e: RuntimeError):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. "
                        + "Consider removing it from the model."
                    )

    def run(self):
        try:
            return self.trainer.train(
                disable_eval_tqdm=self.config.get("show_eval_progressbar", True),
                debug_batches=self.config.get("debug_batches", -1),
            )
        except RuntimeError as e:
            self._process_error(e)
            raise e


@registry.register_task("predict")
class PredictTask(BaseTask):
    def run(self):
        assert (
            self.trainer.test_loader is not None
        ), "Test dataset is required for making predictions"
        assert self.config["checkpoint"]
        results_file = "predictions"
        self.trainer.predict(
            self.trainer.test_loader,
            results_file=results_file,
            disable_tqdm=self.config.get("show_eval_progressbar", True),
        )


@registry.register_task("validate")
class ValidateTask(BaseTask):
    def run(self):
        # Note that the results won't be precise on multi GPUs due to
        # padding of extra images (although the difference should be minor)
        assert (
            "default_val" in self.trainer.config["dataset"]
            and self.trainer.config["dataset"]["default_val"] in self.trainer.loaders
        ), "Val dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.validate(
            split=self.trainer.config["dataset"]["default_val"],
            disable_tqdm=self.config.get("show_eval_progressbar", True),
        )


@registry.register_task("run-relaxations")
class RelaxationTask(BaseTask):
    def run(self):
        assert isinstance(
            self.trainer, ForcesTrainer
        ), "Relaxations are only possible for ForcesTrainer"
        assert (
            self.trainer.relax_dataset is not None
        ), "Relax dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.run_relaxations()
