"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import datetime
import json
import os
import random
import subprocess
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import OCPDataParallel
from ocpmodels.common.logger import TensorboardLogger, WandBLogger
from ocpmodels.common.meter import Meter
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    build_config,
    plot_histogram,
    save_checkpoint,
    warmup_lr_lambda,
)
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer


@registry.register_trainer("base")
class BaseTrainer:
    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        run_dir=None,
        is_debug=False,
        is_vis=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        name="base_trainer",
    ):
        self.name = name
        self.cpu = cpu

        if torch.cuda.is_available() and not self.cpu:
            self.device = local_rank
        else:
            self.device = "cpu"
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available
        if run_dir is None:
            run_dir = os.getcwd()

        timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(
            self.device
        )
        # create directories from master rank only
        distutils.broadcast(timestamp, 0)
        timestamp = datetime.datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        if identifier:
            timestamp += "-{}".format(identifier)

        self.config = {
            "task": task,
            "model": model.pop("name"),
            "model_attributes": model,
            "optim": optimizer,
            "logger": logger,
            "amp": amp,
            "gpus": distutils.get_world_size() if not self.cpu else 0,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp": timestamp,
                "commit": subprocess.check_output(
                    ["git", "describe", "--always"]
                )
                .strip()
                .decode("ascii"),
                "checkpoint_dir": os.path.join(
                    run_dir, "checkpoints", timestamp
                ),
                "results_dir": os.path.join(run_dir, "results", timestamp),
                "logs_dir": os.path.join(run_dir, "logs", logger, timestamp),
            },
        }
        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        if isinstance(dataset, list):
            self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
            if len(dataset) > 2:
                self.config["test_dataset"] = dataset[2]
        else:
            self.config["dataset"] = dataset

        if not is_debug and distutils.is_master():
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

        self.is_debug = is_debug
        self.is_vis = is_vis

        if distutils.is_master():
            print(yaml.dump(self.config, default_flow_style=False))
        self.load()

        self.evaluator = Evaluator(task=name)

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_task()
        self.load_model()
        self.load_criterion()
        self.load_optimizer()
        self.load_extras()

    # Note: this function is now deprecated. We build config outside of trainer.
    # See build_config in ocpmodels.common.utils.py.
    def load_config_from_yaml_and_cmd(self, args):
        self.config = build_config(args)

        # AMP Scaler
        self.scaler = (
            torch.cuda.amp.GradScaler() if self.config["amp"] else None
        )

        # device
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and not self.cpu) else "cpu"
        )

        # Are we just running sanity checks?
        self.is_debug = args.debug
        self.is_vis = args.vis

        # timestamps and directories
        args.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if args.identifier:
            args.timestamp += "-{}".format(args.identifier)

        args.checkpoint_dir = os.path.join("checkpoints", args.timestamp)
        args.results_dir = os.path.join("results", args.timestamp)
        args.logs_dir = os.path.join(
            "logs", self.config["logger"], args.timestamp
        )

        print(yaml.dump(self.config, default_flow_style=False))
        for arg in vars(args):
            print("{:<20}: {}".format(arg, getattr(args, arg)))

        # TODO(abhshkdz): Handle these parameters better. Maybe move to yaml.
        self.config["cmd"] = args.__dict__
        del args

        if not self.is_debug:
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

            # Dump config parameters
            json.dump(
                self.config,
                open(
                    os.path.join(
                        self.config["cmd"]["checkpoint_dir"], "config.json"
                    ),
                    "w",
                ),
            )

    def load_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["cmd"]["seed"]
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_logger(self):
        self.logger = None
        if not self.is_debug and distutils.is_master():
            assert (
                self.config["logger"] is not None
            ), "Specify logger in config"
            self.logger = registry.get_logger_class(self.config["logger"])(
                self.config
            )

    def load_task(self):
        print("### Loading dataset: {}".format(self.config["task"]["dataset"]))
        dataset = registry.get_dataset_class(self.config["task"]["dataset"])(
            self.config["dataset"]
        )

        if self.config["task"]["dataset"] in ["qm9", "dogss"]:
            num_targets = dataset.data.y.shape[-1]
            if (
                "label_index" in self.config["task"]
                and self.config["task"]["label_index"] is not False
            ):
                dataset.data.y = dataset.data.y[
                    :, int(self.config["task"]["label_index"])
                ]
                num_targets = 1
        else:
            num_targets = 1

        self.num_targets = num_targets
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = dataset.get_dataloaders(
            batch_size=int(self.config["optim"]["batch_size"])
        )

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        if self.config["dataset"].get("normalize_labels", True):
            self.normalizers["target"] = Normalizer(
                self.train_loader.dataset.data.y[
                    self.train_loader.dataset.__indices__
                ],
                self.device,
            )

        # If we're computing gradients wrt input, set mean of normalizer to 0 --
        # since it is lost when compute dy / dx -- and std to forward target std
        if "grad_input" in self.config["task"]:
            if self.config["dataset"].get("normalize_labels", True):
                self.normalizers["grad_target"] = Normalizer(
                    self.train_loader.dataset.data.y[
                        self.train_loader.dataset.__indices__
                    ],
                    self.device,
                )
                self.normalizers["grad_target"].mean.fill_(0)

        if self.is_vis and self.config["task"]["dataset"] != "qm9":
            # Plot label distribution.
            plots = [
                plot_histogram(
                    self.train_loader.dataset.data.y.tolist(),
                    xlabel="{}/raw".format(self.config["task"]["labels"][0]),
                    ylabel="# Examples",
                    title="Split: train",
                ),
                plot_histogram(
                    self.val_loader.dataset.data.y.tolist(),
                    xlabel="{}/raw".format(self.config["task"]["labels"][0]),
                    ylabel="# Examples",
                    title="Split: val",
                ),
                plot_histogram(
                    self.test_loader.dataset.data.y.tolist(),
                    xlabel="{}/raw".format(self.config["task"]["labels"][0]),
                    ylabel="# Examples",
                    title="Split: test",
                ),
            ]
            self.logger.log_plots(plots)

    def load_model(self):
        # Build model
        if distutils.is_master():
            print("### Loading model: {}".format(self.config["model"]))

        # TODO(abhshkdz): Eventually move towards computing features on-the-fly
        # and remove dependence from `.edge_attr`.
        bond_feat_dim = None
        if self.config["task"]["dataset"] in [
            "trajectory_lmdb",
            "single_point_lmdb",
        ]:
            bond_feat_dim = self.config["model_attributes"].get(
                "num_gaussians", 50
            )
        else:
            raise NotImplementedError

        self.model = registry.get_model_class(self.config["model"])(
            self.train_loader.dataset[0].x.shape[-1]
            if hasattr(self.train_loader.dataset[0], "x")
            and self.train_loader.dataset[0].x is not None
            else None,
            bond_feat_dim,
            self.num_targets,
            **self.config["model_attributes"],
        )  # .to(self.device)

        if distutils.is_master():
            print(
                "### Loaded {} with {} parameters.".format(
                    self.model.__class__.__name__, self.model.num_params
                )
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        #        self.model = OCPDataParallel(
        #            self.model,
        #            output_device=self.device,
        #            num_gpus=1 if not self.cpu else 0,
        #        )
        if distutils.initialized():
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )

    def load_pretrained(self, checkpoint_path=None, ddp_to_dp=False):
        if checkpoint_path is None or os.path.isfile(checkpoint_path) is False:
            print(f"Checkpoint: {checkpoint_path} not found!")
            return False

        print("### Loading checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        # Load model, optimizer, normalizer state dict.
        # if trained with ddp and want to load in non-ddp, modify keys from
        # module.module.. -> module..
        if ddp_to_dp:
            new_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:]
                new_dict[name] = v
            self.model.load_state_dict(new_dict)
        else:
            self.model.load_state_dict(checkpoint["state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])

        for key in checkpoint["normalizers"]:
            if key in self.normalizers:
                self.normalizers[key].load_state_dict(
                    checkpoint["normalizers"][key]
                )
            if self.scaler and checkpoint["amp"]:
                self.scaler.load_state_dict(checkpoint["amp"])
        return True

    # TODO(abhshkdz): Rename function to something nicer.
    # TODO(abhshkdz): Support multiple loss functions.
    def load_criterion(self):
        self.criterion = self.config["optim"].get("criterion", nn.L1Loss())

    def load_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            self.config["optim"]["lr_initial"],  # weight_decay=3.0
        )

    def load_extras(self):
        # learning rate scheduler.
        scheduler_lambda_fn = lambda x: warmup_lr_lambda(
            x, self.config["optim"]
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=scheduler_lambda_fn
        )

        # metrics.
        self.meter = Meter(split="train")

    def save(self, epoch, metrics):
        if not self.is_debug and distutils.is_master():
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "normalizers": {
                        key: value.state_dict()
                        for key, value in self.normalizers.items()
                    },
                    "config": self.config,
                    "val_metrics": metrics,
                    "amp": self.scaler.state_dict() if self.scaler else None,
                },
                self.config["cmd"]["checkpoint_dir"],
            )

    def train(self, max_epochs=None, return_metrics=False):
        # TODO(abhshkdz): Timers for dataloading and forward pass.
        num_epochs = (
            max_epochs
            if max_epochs is not None
            else self.config["optim"]["max_epochs"]
        )
        for epoch in range(num_epochs):
            self.model.train()

            for i, batch in enumerate(self.train_loader):
                batch = batch.to(self.device)

                # Forward, loss, backward.
                out, metrics = self._forward(batch)
                loss = self._compute_loss(out, batch)
                self._backward(loss)

                # Update meter.
                meter_update_dict = {
                    "epoch": epoch + (i + 1) / len(self.train_loader),
                    "loss": loss.item(),
                }
                meter_update_dict.update(metrics)
                self.meter.update(meter_update_dict)

                # Make plots.
                if self.logger is not None:
                    self.logger.log(
                        meter_update_dict,
                        step=epoch * len(self.train_loader) + i + 1,
                        split="train",
                    )

                # Print metrics.
                if i % self.config["cmd"]["print_every"] == 0:
                    print(self.meter)

            self.scheduler.step()

            if self.val_loader is not None:
                v_loss, v_mae = self.validate(split="val", epoch=epoch)

            if self.test_loader is not None:
                test_loss, test_mae = self.validate(split="test", epoch=epoch)

            if not self.is_debug:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "amp": self.scaler.state_dict()
                        if self.scaler
                        else None,
                    },
                    self.config["cmd"]["checkpoint_dir"],
                )
        if return_metrics:
            return {
                "training_loss": float(self.meter.loss.global_avg),
                "training_mae": float(
                    self.meter.meters[
                        self.config["task"]["labels"][0]
                        + "/"
                        + self.config["task"]["metric"]
                    ].global_avg
                ),
                "validation_loss": v_loss,
                "validation_mae": v_mae,
                "test_loss": test_loss,
                "test_mae": test_mae,
            }

    @torch.no_grad()
    def validate(self, split="val", epoch=None):
        if distutils.is_master():
            print("### Evaluating on {}.".format(split))

        self.model.eval()
        evaluator, metrics = Evaluator(task=self.name), {}
        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
        ):
            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)

            if self.config["task"].get("pipeline_parallel", False):
                # loss = self._compute_loss(out, [batch[0].to("cuda:7")])
                loss = self._compute_loss(out, [self.to_last_device(batch[0])])
            else:
                loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": epoch + 1})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            print(", ".join(log_str))

        # Make plots.
        if self.logger is not None and epoch is not None:
            self.logger.log(
                log_dict,
                step=(epoch + 1) * len(self.train_loader),
                split=split,
            )

        return metrics

    def _forward(self, batch, compute_metrics=True):
        out = {}

        # enable gradient wrt input.
        if "grad_input" in self.config["task"]:
            inp_for_grad = batch.pos
            batch.pos = batch.pos.requires_grad_(True)

        # forward pass.
        if self.config["model_attributes"].get("regress_forces", False):
            output, output_forces = self.model(batch)
        else:
            output = self.model(batch)

        if output.shape[-1] == 1:
            output = output.view(-1)

        out["output"] = output

        force_output = None
        if self.config["model_attributes"].get("regress_forces", False):
            out["force_output"] = output_forces
            force_output = output_forces

        if (
            "grad_input" in self.config["task"]
            and self.config["model_attributes"].get("regress_forces", False)
            is False
        ):
            force_output = -1 * torch.autograd.grad(
                output,
                inp_for_grad,
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True,
            )[0]
            out["force_output"] = force_output

        if not compute_metrics:
            return out, None

        metrics = {}

        if self.config["dataset"].get("normalize_labels", True):
            errors = eval(self.config["task"]["metric"])(
                self.normalizers["target"].denorm(output).cpu(), batch.y.cpu()
            ).view(-1)
        else:
            errors = eval(self.config["task"]["metric"])(
                output.cpu(), batch.y.cpu()
            ).view(-1)

        if (
            "label_index" in self.config["task"]
            and self.config["task"]["label_index"] is not False
        ):
            # TODO(abhshkdz): Get rid of this edge case for QM9.
            # This is only because QM9 has multiple targets and we can either
            # jointly predict all of them or one particular target.
            metrics[
                "{}/{}".format(
                    self.config["task"]["labels"][
                        self.config["task"]["label_index"]
                    ],
                    self.config["task"]["metric"],
                )
            ] = errors[0]
        else:
            for i, label in enumerate(self.config["task"]["labels"]):
                metrics[
                    "{}/{}".format(label, self.config["task"]["metric"])
                ] = errors[i]

        if "grad_input" in self.config["task"]:
            force_pred = force_output
            force_target = batch.force

            if self.config["task"].get("eval_on_free_atoms", True):
                mask = batch.fixed == 0
                force_pred = force_pred[mask]
                force_target = force_target[mask]

            if self.config["dataset"].get("normalize_labels", True):
                grad_input_errors = eval(self.config["task"]["metric"])(
                    self.normalizers["grad_target"].denorm(force_pred).cpu(),
                    force_target.cpu(),
                )
            else:
                grad_input_errors = eval(self.config["task"]["metric"])(
                    force_pred.cpu(), force_target.cpu()
                )
            metrics[
                "force_x/{}".format(self.config["task"]["metric"])
            ] = grad_input_errors[0]
            metrics[
                "force_y/{}".format(self.config["task"]["metric"])
            ] = grad_input_errors[1]
            metrics[
                "force_z/{}".format(self.config["task"]["metric"])
            ] = grad_input_errors[2]

        return out, metrics

    def _compute_loss(self, out, batch):
        loss = []

        if self.config["dataset"].get("normalize_labels", True):
            target_normed = self.normalizers["target"].norm(batch.y)
        else:
            target_normed = batch.y

        loss.append(self.criterion(out["output"], target_normed))

        # TODO(abhshkdz): Test support for gradients wrt input.
        # TODO(abhshkdz): Make this general; remove dependence on `.forces`.
        if "grad_input" in self.config["task"]:
            if self.config["dataset"].get("normalize_labels", True):
                grad_target_normed = self.normalizers["grad_target"].norm(
                    batch.force
                )
            else:
                grad_target_normed = batch.force

            # Force coefficient = 30 has been working well for us.
            force_mult = self.config["optim"].get("force_coefficient", 30)
            if self.config["task"].get("train_on_free_atoms", False):
                mask = batch.fixed == 0
                loss.append(
                    force_mult
                    * self.criterion(
                        out["force_output"][mask], grad_target_normed[mask]
                    )
                )
            else:
                loss.append(
                    force_mult
                    * self.criterion(out["force_output"], grad_target_normed)
                )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # TODO(abhshkdz): Add support for gradient clipping.
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def save_results(self, predictions, results_file, keys):
        if results_file is None:
            return

        results_file_path = os.path.join(
            self.config["cmd"]["results_dir"],
            f"{self.name}_{results_file}_{distutils.get_rank()}.npz",
        )
        np.savez_compressed(
            results_file_path,
            ids=predictions["id"],
            **{key: predictions[key] for key in keys},
        )

        distutils.synchronize()
        if distutils.is_master():
            gather_results = defaultdict(list)
            full_path = os.path.join(
                self.config["cmd"]["results_dir"],
                f"{self.name}_{results_file}.npz",
            )

            for i in range(distutils.get_world_size()):
                rank_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    f"{self.name}_{results_file}_{i}.npz",
                )
                rank_results = np.load(rank_path, allow_pickle=True)
                gather_results["ids"].extend(rank_results["ids"])
                for key in keys:
                    gather_results[key].extend(rank_results[key])
                os.remove(rank_path)

            # Because of how distributed sampler works, some system ids
            # might be repeated to make no. of samples even across GPUs.
            _, idx = np.unique(gather_results["ids"], return_index=True)
            gather_results["ids"] = np.array(gather_results["ids"])[idx]
            for k in keys:
                if k == "forces":
                    gather_results[k] = np.array(
                        gather_results[k], dtype=object
                    )[idx]
                else:
                    gather_results[k] = np.array(gather_results[k])[idx]

            print(f"Writing results to {full_path}")
            np.savez_compressed(full_path, **gather_results)

    def to_last_device(self, batch):
        """
        Helper function to move a batch to the last device for pipeline
        parallel usecase.

        """
        assert (
            self.config["model_attributes"]["pipeline_parallel_devices"] != []
        )
        batch = batch.to(
            self.config["model_attributes"]["pipeline_parallel_devices"][-1]
        )
        return batch
