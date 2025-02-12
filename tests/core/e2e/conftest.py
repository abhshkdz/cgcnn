from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def configs():
    return {
        "scn": Path("tests/core/models/test_configs/test_scn.yml"),
        "escn": Path("tests/core/models/test_configs/test_escn.yml"),
        "escn_hydra": Path("tests/core/models/test_configs/test_escn_hydra.yml"),
        "schnet": Path("tests/core/models/test_configs/test_schnet.yml"),
        "gemnet_dt": Path("tests/core/models/test_configs/test_gemnet_dt.yml"),
        "gemnet_dt_hydra": Path(
            "tests/core/models/test_configs/test_gemnet_dt_hydra.yml"
        ),
        "gemnet_dt_hydra_grad": Path(
            "tests/core/models/test_configs/test_gemnet_dt_hydra_grad.yml"
        ),
        "gemnet_oc": Path("tests/core/models/test_configs/test_gemnet_oc.yml"),
        "gemnet_oc_hydra": Path(
            "tests/core/models/test_configs/test_gemnet_oc_hydra.yml"
        ),
        "gemnet_oc_hydra_grad": Path(
            "tests/core/models/test_configs/test_gemnet_oc_hydra_grad.yml"
        ),
        "dimenet++": Path("tests/core/models/test_configs/test_dpp.yml"),
        "dimenet++_hydra": Path("tests/core/models/test_configs/test_dpp_hydra.yml"),
        "painn": Path("tests/core/models/test_configs/test_painn.yml"),
        "painn_hydra": Path("tests/core/models/test_configs/test_painn_hydra.yml"),
        "equiformer_v2": Path("tests/core/models/test_configs/test_equiformerv2.yml"),
        "equiformer_v2_hydra": Path(
            "tests/core/models/test_configs/test_equiformerv2_hydra.yml"
        ),
        "escaip_hydra": Path("tests/core/models/test_configs/test_escaip_hydra.yml"),
        "escaip_hydra_grad": Path(
            "tests/core/models/test_configs/test_escaip_hydra_grad.yml"
        ),
    }


@pytest.fixture()
def tutorial_train_src(tutorial_dataset_path):
    return tutorial_dataset_path / "s2ef/train_100"


@pytest.fixture()
def tutorial_val_src(tutorial_dataset_path):
    return tutorial_dataset_path / "s2ef/val_20"
