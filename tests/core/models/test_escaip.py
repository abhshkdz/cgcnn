from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import yaml
from ase.io import read
from test_rank2_head import _reshape_tensor

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import setup_imports
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing import AtomsToGraphs

skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="skipping when no gpu"
)


@pytest.fixture(scope="class")
def load_data(request):
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=0,
        format="json",
    )
    a2g = AtomsToGraphs(
        max_neigh=200,
        radius=6,
        r_edges=False,
        r_fixed=True,
    )
    data_list = a2g.convert_all([atoms])
    request.cls.data = data_list[0]


def _load_hydra_model():
    torch.manual_seed(4)
    with open(
        Path("tests/core/models/test_configs/test_escaip_hydra.yml")
    ) as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
        # add stress head
        yaml_config["model"]["heads"]["stress"] = {"module": "EScAIP_rank2_head"}
        model = registry.get_model_class("hydra")(
            yaml_config["model"]["backbone"],
            yaml_config["model"]["heads"],
            pass_through_head_outputs=True,
        )
    model.backbone.num_layers = 1
    return model


@pytest.fixture(scope="class")
def load_model(request):
    setup_imports()
    request.cls.model = _load_hydra_model()


@pytest.mark.usefixtures("load_data")
@pytest.mark.usefixtures("load_model")
class TestEquiformerV2:
    def test_energy_force_stress_shape(self):
        data = self.data
        model = self.model
        model.eval()
        with torch.no_grad():
            output = model(data_list_collater([data]))
        print(output)
        assert output["energy"].shape[-1] == (1)
        assert output["forces"].shape[-1] == (3)
        assert output["stress_isotropic"].shape[-1] == (1)
        assert output["stress_anisotropic"].shape[-1] == (5)

        tensor = _reshape_tensor(output["stress_isotropic"])
        # anisotropic must be traceless
        assert torch.diagonal(tensor).sum().item() == pytest.approx(0.0, abs=2e-8)
        # all tensors must be symmetric
        assert torch.allclose(tensor, tensor.transpose(0, 1))
