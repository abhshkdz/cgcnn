import numpy as np
import pytest

from fairchem.core.datasets.base_dataset import create_dataset
from fairchem.core.scripts.make_lmdb_sizes import get_lmdb_sizes_parser, make_lmdb_sizes


@pytest.fixture()
def config(tutorial_dataset_path):
    lmdb_path = str(tutorial_dataset_path / "s2ef/val_20")

    # make dataset metadata
    parser = get_lmdb_sizes_parser()
    args, override_args = parser.parse_known_args(["--data-path", lmdb_path])
    make_lmdb_sizes(args)

    config = {
        "format": "lmdb",
        "src": lmdb_path,
    }
    return config


def test_load_lmdb_dataset(config):
    dataset = create_dataset(config, split="val")

    assert dataset.get_metadata("natoms", 0) == dataset[0].natoms

    all_metadata_natoms = np.array(dataset.get_metadata("natoms", range(len(dataset))))
    all_natoms = np.array([datapoint.natoms for datapoint in dataset])

    assert (all_natoms == all_metadata_natoms).all()
