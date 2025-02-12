from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
from test_e2e_commons import _run_main, oc20_lmdb_train_and_val_from_paths

from fairchem.core.scripts.convert_hydra_to_release import convert_fine_tune_checkpoint


@pytest.fixture()
def tutorial_val_src(tutorial_dataset_path):
    return tutorial_dataset_path / "s2ef/val_20"


def make_checkpoint(tempdir: str, data_source: Path, seed: int) -> str:
    # first train a tiny eqv2 model to get a checkpoint
    eqv2_yml = Path("tests/core/models/test_configs/test_equiformerv2_hydra.yml")
    ck_path = os.path.join(tempdir, "checkpoint.pt")
    _run_main(
        tempdir,
        eqv2_yml,
        update_dict_with={
            "optim": {
                "max_epochs": 1,
                "eval_every": 8,
                "batch_size": 1,
                "num_workers": 0,
            },
            "dataset": oc20_lmdb_train_and_val_from_paths(
                train_src=str(data_source),
                val_src=str(data_source),
                test_src=str(data_source),
                otf_norms=False,
            ),
        },
        update_run_args_with={"seed": seed},
        save_checkpoint_to=ck_path,
        world_size=1,
    )
    assert os.path.isfile(ck_path)
    return ck_path


def run_main_with_ft_hydra(
    tempdir: str,
    yaml: str,
    data_src: str,
    run_args: dict,
    model_config: str,
    output_checkpoint: str,
):
    _run_main(
        tempdir,
        yaml,
        update_dict_with={
            "optim": {
                "max_epochs": 1,
                "eval_every": 8,
                "batch_size": 1,
                "num_workers": 0,
                "lr_initial": 0.0,  # don't learn anything
            },
            "dataset": oc20_lmdb_train_and_val_from_paths(
                train_src=str(data_src),
                val_src=str(data_src),
                test_src=str(data_src),
                otf_norms=False,
            ),
            "model": model_config,
        },
        update_run_args_with=run_args,
        save_checkpoint_to=output_checkpoint,
        world_size=1,
    )


def verify_release_checkpoint(release_yaml_fn, release_checkpoint_fn, ft_state_dict):
    with tempfile.TemporaryDirectory() as temp_dir:
        # now lets run the new release checkpoint for a few iterations at lr0.0
        ck_release_ft_afterload_path = os.path.join(
            temp_dir, "checkpoint_ft_release.pt"
        )
        release_ft_temp_dir = os.path.join(temp_dir, "release_ft")
        os.makedirs(release_ft_temp_dir)

        _run_main(
            release_ft_temp_dir,
            release_yaml_fn,
            update_run_args_with={"seed": 1337, "checkpoint": release_checkpoint_fn},
            save_checkpoint_to=ck_release_ft_afterload_path,
            world_size=1,
            update_dict_with={
                "optim": {
                    "max_epochs": 2,
                }
            },
        )

        # make sure the checkpoint after running with lr0.0 is identical
        # to the previous checkpoint
        assert os.path.isfile(ck_release_ft_afterload_path)
        ft_after_state_dict = torch.load(ck_release_ft_afterload_path)["state_dict"]
        for key in ft_after_state_dict:
            if (
                key.startswith("module.backbone")
                or key.startswith("module.output_heads")
                and key.endswith("weight")
            ):
                assert torch.allclose(ft_after_state_dict[key], ft_state_dict[key])


def test_finetune_hydra_freeze_backbone(tutorial_val_src):
    with tempfile.TemporaryDirectory() as orig_ckpt_dir:
        starting_ckpt = make_checkpoint(orig_ckpt_dir, tutorial_val_src, 0)
        old_state_dict = torch.load(starting_ckpt)["state_dict"]

        # Test to make sure without freeze the backbone weights change
        with tempfile.TemporaryDirectory() as ft_temp_dir:
            ft_yml = Path("tests/core/models/test_configs/test_finetune_hydra.yml")
            ck_ft_path = os.path.join(ft_temp_dir, "checkpoint_ft.pt")
            model_config = {
                "name": "hydra",
                "finetune_config": {"starting_checkpoint": starting_ckpt},
                "heads": {
                    "energy": {"module": "equiformer_v2_energy_head"},
                    "forces": {"module": "equiformer_v2_force_head"},
                },
            }

            _run_main(
                ft_temp_dir,
                ft_yml,
                update_dict_with={
                    "optim": {
                        "max_epochs": 1,
                        "eval_every": 8,
                        "batch_size": 1,
                        "num_workers": 0,
                        "lr_initial": 10.0,
                    },
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                        otf_norms=False,
                    ),
                    "model": model_config,
                },
                update_run_args_with={"seed": 1000},
                save_checkpoint_to=ck_ft_path,
                world_size=1,
            )

            assert os.path.isfile(ck_ft_path)
            ft_ckpt = torch.load(ck_ft_path)
            assert "config" in ft_ckpt
            assert ft_ckpt["config"]["model"]["name"] == "hydra"
            # check that the backbone weights are different, and other weights are not the same
            ft_state_dict = ft_ckpt["state_dict"]
            for key in ft_state_dict:
                if key.startswith("module.backbone") and ".weight" in key:
                    # backbone should be different
                    assert not torch.allclose(ft_state_dict[key], old_state_dict[key])
                elif key.startswith("module.output_heads") and key.endswith("weight"):
                    # heads weight should be different because the seeds are different
                    assert not torch.allclose(ft_state_dict[key], old_state_dict[key])

        # Test to make sure with freeze the backbone weights are unchanged
        with tempfile.TemporaryDirectory() as ft_temp_dir:
            ft_yml = Path("tests/core/models/test_configs/test_finetune_hydra.yml")
            ck_ft_path = os.path.join(ft_temp_dir, "checkpoint_ft.pt")
            model_config = {
                "name": "hydra",
                "finetune_config": {"starting_checkpoint": starting_ckpt},
                "heads": {
                    "energy": {"module": "equiformer_v2_energy_head"},
                    "forces": {"module": "equiformer_v2_force_head"},
                },
                "freeze_backbone": True,
            }

            _run_main(
                ft_temp_dir,
                ft_yml,
                update_dict_with={
                    "optim": {
                        "max_epochs": 1,
                        "eval_every": 8,
                        "batch_size": 1,
                        "num_workers": 0,
                    },
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                        otf_norms=False,
                    ),
                    "model": model_config,
                },
                update_run_args_with={"seed": 1000},
                save_checkpoint_to=ck_ft_path,
                world_size=1,
            )

            assert os.path.isfile(ck_ft_path)
            ft_ckpt = torch.load(ck_ft_path)
            assert "config" in ft_ckpt
            assert ft_ckpt["config"]["model"]["name"] == "hydra"
            # check that the backbone weights are different, and other weights are not the same
            ft_state_dict = ft_ckpt["state_dict"]
            for key in ft_state_dict:
                if key.startswith("module.backbone"):
                    # backbone should be different
                    assert torch.allclose(ft_state_dict[key], old_state_dict[key])
                elif key.startswith("module.output_heads") and key.endswith("weight"):
                    # heads weight should be different because the seeds are different
                    assert not torch.allclose(ft_state_dict[key], old_state_dict[key])


def test_finetune_hydra_retain_backbone(tutorial_val_src):
    with tempfile.TemporaryDirectory() as orig_ckpt_dir:
        starting_ckpt = make_checkpoint(orig_ckpt_dir, tutorial_val_src, 0)
        old_state_dict = torch.load(starting_ckpt)["state_dict"]
        # now finetune a the model with the checkpoint from the first job
        with tempfile.TemporaryDirectory() as ft_temp_dir:
            ft_yml = Path("tests/core/models/test_configs/test_finetune_hydra.yml")
            ck_ft_path = os.path.join(ft_temp_dir, "checkpoint_ft.pt")
            model_config = {
                "name": "hydra",
                "finetune_config": {"starting_checkpoint": starting_ckpt},
                "heads": {
                    "energy": {"module": "equiformer_v2_energy_head"},
                    "forces": {"module": "equiformer_v2_force_head"},
                },
            }
            run_main_with_ft_hydra(
                tempdir=ft_temp_dir,
                yaml=ft_yml,
                data_src=tutorial_val_src,
                run_args={"seed": 1000},
                model_config=model_config,
                output_checkpoint=ck_ft_path,
            )
            assert os.path.isfile(ck_ft_path)
            ft_ckpt = torch.load(ck_ft_path)
            assert "config" in ft_ckpt
            assert ft_ckpt["config"]["model"]["name"] == "hydra"
            # check that the backbone weights are the same, and other weights are not the same
            ft_state_dict = ft_ckpt["state_dict"]
            for key in ft_state_dict:
                if key.startswith("module.backbone"):
                    # backbone should be identical
                    assert torch.allclose(ft_state_dict[key], old_state_dict[key])
                elif key.startswith("module.output_heads") and key.endswith("weight"):
                    # heads weight should be different because the seeds are different
                    assert not torch.allclose(ft_state_dict[key], old_state_dict[key])

            # Add a test to convert the FT hydra checkpoint to a release checkpoint
            # This could be a separate test but we would need to generate the FT checkpoint
            # all over again

            # Convert FT hydra checkpoint to release checkpoint
            ck_release_ft_path = os.path.join(ft_temp_dir, "checkpoint_ft_release.pt")
            yml_release_ft_path = os.path.join(ft_temp_dir, "checkpoint_ft_release.yml")
            # the actual on disk yaml used in the previous run, after argument updates
            fine_tune_yaml_fn = os.path.join(ft_temp_dir, "test_run.yml")
            convert_fine_tune_checkpoint(
                fine_tune_checkpoint_fn=ck_ft_path,
                fine_tune_yaml_fn=fine_tune_yaml_fn,
                output_checkpoint_fn=ck_release_ft_path,
                output_yaml_fn=yml_release_ft_path,
            )

            # remove starting checkpoint, so that we cant accidentally load it
            os.remove(ck_ft_path)

            verify_release_checkpoint(
                yml_release_ft_path, ck_release_ft_path, ft_state_dict
            )


def test_finetune_hydra_override(tutorial_val_src):
    with tempfile.TemporaryDirectory() as orig_ckpt_dir:
        starting_ckpt = make_checkpoint(orig_ckpt_dir, tutorial_val_src, 0)
        # now finetune a the model with the checkpoint from the first job
        with tempfile.TemporaryDirectory() as ft_temp_dir:
            ft_yml = Path("tests/core/models/test_configs/test_finetune_hydra.yml")
            ck_ft_path = os.path.join(ft_temp_dir, "checkpoint_ft.pt")
            model_config = {
                "name": "hydra",
                "finetune_config": {
                    "starting_checkpoint": starting_ckpt,
                    "override": {"forward": None},
                },
                "heads": {
                    "energy": {"module": "equiformer_v2_energy_head"},
                    "forces": {"module": "equiformer_v2_force_head"},
                },
            }

            # TODO add a better test for override when we get there
            # for now just override .forward() with None
            with pytest.raises(TypeError):
                run_main_with_ft_hydra(
                    tempdir=ft_temp_dir,
                    yaml=ft_yml,
                    data_src=tutorial_val_src,
                    run_args={"seed": 1000},
                    model_config=model_config,
                    output_checkpoint=ck_ft_path,
                )


def test_finetune_hydra_data_only(tutorial_val_src):
    with tempfile.TemporaryDirectory() as orig_ckpt_dir:
        starting_ckpt = make_checkpoint(orig_ckpt_dir, tutorial_val_src, 0)
        old_state_dict = torch.load(starting_ckpt)["state_dict"]
        # now finetune a the model with the checkpoint from the first job
        with tempfile.TemporaryDirectory() as ft_temp_dir:
            ft_yml = Path("tests/core/models/test_configs/test_finetune_hydra.yml")
            ck_ft_path = os.path.join(ft_temp_dir, "checkpoint_ft.pt")
            model_config = {
                "name": "hydra",
                "finetune_config": {"starting_checkpoint": starting_ckpt},
            }
            run_main_with_ft_hydra(
                tempdir=ft_temp_dir,
                yaml=ft_yml,
                data_src=tutorial_val_src,
                run_args={"seed": 1000},
                model_config=model_config,
                output_checkpoint=ck_ft_path,
            )
            assert os.path.isfile(ck_ft_path)
            ft_ckpt = torch.load(ck_ft_path)
            assert "config" in ft_ckpt
            config_model = ft_ckpt["config"]["model"]
            assert config_model["name"] == "hydra"
            # check that the entire model weights are the same
            ft_state_dict = ft_ckpt["state_dict"]
            assert len(ft_state_dict) == len(old_state_dict)
            for key in ft_state_dict:
                assert torch.allclose(ft_state_dict[key], old_state_dict[key])

            # Add a test to convert the FT hydra checkpoint to a release checkpoint
            # This could be a separate test but we would need to generate the FT checkpoint
            # all over again

            # Convert FT hydra checkpoint to release checkpoint
            ck_release_ft_path = os.path.join(ft_temp_dir, "checkpoint_ft_release.pt")
            yml_release_ft_path = os.path.join(ft_temp_dir, "checkpoint_ft_release.yml")
            # the actual on disk yaml used in the previous run, after argument updates
            fine_tune_yaml_fn = os.path.join(ft_temp_dir, "test_run.yml")
            convert_fine_tune_checkpoint(
                fine_tune_checkpoint_fn=ck_ft_path,
                fine_tune_yaml_fn=fine_tune_yaml_fn,
                output_checkpoint_fn=ck_release_ft_path,
                output_yaml_fn=yml_release_ft_path,
            )

            # remove starting checkpoint, so that we cant accidentally load it
            os.remove(ck_ft_path)

            verify_release_checkpoint(
                yml_release_ft_path, ck_release_ft_path, ft_state_dict
            )


def test_finetune_from_finetunehydra(tutorial_val_src):
    with tempfile.TemporaryDirectory() as orig_ckpt_dir:
        starting_ckpt = make_checkpoint(orig_ckpt_dir, tutorial_val_src, 0)
        # now finetune a the model with the checkpoint from the first job
        with tempfile.TemporaryDirectory() as finetune_run1_dir:
            ft_yml = Path("tests/core/models/test_configs/test_finetune_hydra.yml")
            ck_ft_path = os.path.join(finetune_run1_dir, "checkpoint_ft.pt")
            model_config_1 = {
                "name": "hydra",
                "finetune_config": {"starting_checkpoint": starting_ckpt},
            }
            run_main_with_ft_hydra(
                tempdir=finetune_run1_dir,
                yaml=ft_yml,
                data_src=tutorial_val_src,
                run_args={"seed": 1000},
                model_config=model_config_1,
                output_checkpoint=ck_ft_path,
            )
            assert os.path.isfile(ck_ft_path)

            # now that we have a second checkpoint, try finetuning again from this checkpoint
            ########################################################################################
            with tempfile.TemporaryDirectory() as finetune_run2_dir:
                ck_ft2_path = os.path.join(finetune_run2_dir, "checkpoint_ft.pt")
                model_config_2 = {
                    "name": "hydra",
                    "finetune_config": {"starting_checkpoint": ck_ft_path},
                }
                run_main_with_ft_hydra(
                    tempdir=finetune_run2_dir,
                    yaml=ft_yml,
                    data_src=tutorial_val_src,
                    run_args={"seed": 1000},
                    model_config=model_config_2,
                    output_checkpoint=ck_ft2_path,
                )
                ft_ckpt2 = torch.load(ck_ft2_path)
                assert "config" in ft_ckpt2
                config_model = ft_ckpt2["config"]["model"]
                assert config_model["name"] == "hydra"
                old_state_dict = torch.load(ck_ft_path)["state_dict"]
                new_state_dict = ft_ckpt2["state_dict"]
                # the state dicts should still be identical because we made the LR = 0.0
                for key in new_state_dict:
                    assert torch.allclose(new_state_dict[key], old_state_dict[key])
