from __future__ import annotations

import os

import numpy as np
import pytest
from ase import build, db
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import Trajectory, write

from fairchem.core.datasets import (
    AseDBDataset,
    AseReadDataset,
    AseReadMultiStructureDataset,
)


@pytest.fixture(
    params=[
        "db_dataset",
        "db_dataset_folder",
        "db_dataset_list",
        "db_dataset_path_list",
        "lmdb_dataset",
        "aselmdb_dataset",
    ],
)
def ase_dataset(request, structures, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dataset")
    mult = 1
    a2g_args = {
        "r_energy": True,
        "r_forces": True,
        "r_stress": True,
        "r_data_keys": ["extensive_property", "tensor_property"],
    }
    if request.param == "db_dataset":
        with db.connect(tmp_path / "asedb.db") as database:
            for _i, atoms in enumerate(structures):
                database.write(atoms, data=atoms.info)
        dataset = AseDBDataset(
            config={"src": str(tmp_path / "asedb.db"), "a2g_args": a2g_args}
        )
    elif request.param == "db_dataset_folder" or request.param == "db_dataset_list":
        for db_name in ("asedb1.db", "asedb2.db"):
            with db.connect(tmp_path / db_name) as database:
                for _i, atoms in enumerate(structures):
                    database.write(atoms, data=atoms.info)
        mult = 2
        src = (
            str(tmp_path)
            if request.param == "db_dataset_folder"
            else [str(tmp_path / "asedb1.db"), str(tmp_path / "asedb2.db")]
        )
        dataset = AseDBDataset(config={"src": src, "a2g_args": a2g_args})
    elif request.param == "db_dataset_path_list":
        os.mkdir(tmp_path / "dir1")
        os.mkdir(tmp_path / "dir2")

        for dir_name in ("dir1", "dir2"):
            for db_name in ("asedb1.db", "asedb2.db"):
                with db.connect(tmp_path / dir_name / db_name) as database:
                    for _i, atoms in enumerate(structures):
                        database.write(atoms, data=atoms.info)
        mult = 4
        dataset = AseDBDataset(
            config={
                "src": [str(tmp_path / "dir1"), str(tmp_path / "dir2")],
                "a2g_args": a2g_args,
            }
        )
    elif request.param == "lmdb_dataset":
        with connect(str(tmp_path / "asedb.aselmdb")) as database:
            for _i, atoms in enumerate(structures):
                database.write(atoms, data=atoms.info)

        dataset = AseDBDataset(
            config={"src": str(tmp_path / "asedb.aselmdb"), "a2g_args": a2g_args}
        )
    else:  # "aselmbd_dataset" with .aselmdb file extension
        with connect(str(tmp_path / "asedb.aselmdb")) as database:
            for _i, atoms in enumerate(structures):
                database.write(atoms, data=atoms.info)

        dataset = AseDBDataset(
            config={"src": str(tmp_path / "asedb.aselmdb"), "a2g_args": a2g_args}
        )

    return dataset, mult


def test_ase_dataset(ase_dataset, structures):
    dataset, mult = ase_dataset
    assert len(dataset) == mult * len(structures)
    for data in dataset:
        assert hasattr(data, "y")
        assert data.forces.shape == (data.natoms, 3)
        assert data.stress.shape == (3, 3)
        assert data.tensor_property.shape == (6, 6)
        assert isinstance(data.extensive_property, int)


def test_ase_read_dataset(tmp_path, structures):
    # unfortunately there is currently no clean (already implemented) way to save atoms.info when saving
    # individual structures - so test separately
    for i, structure in enumerate(structures):
        write(tmp_path / f"{i}.cif", structure)

    dataset = AseReadDataset(
        config={
            "src": str(tmp_path),
            "pattern": "*.cif",
        }
    )

    assert len(dataset) == len(structures)
    data = dataset[0]
    del data


def test_ase_get_metadata(ase_dataset):
    assert ase_dataset[0].get_metadata("natoms", [0])[0] == 3


def test_ase_metadata_guesser(ase_dataset):
    dataset, _ = ase_dataset

    metadata = dataset.sample_property_metadata()

    # Confirm energy metadata guessed properly
    assert metadata["targets"]["energy"]["extensive"] is False
    assert metadata["targets"]["energy"]["shape"] == ()
    assert metadata["targets"]["energy"]["type"] == "per-image"

    # Confirm forces metadata guessed properly
    assert metadata["targets"]["forces"]["shape"] == (3,)
    assert metadata["targets"]["forces"]["extensive"] is True
    assert metadata["targets"]["forces"]["type"] == "per-atom"

    # Confirm stress metadata guessed properly
    assert metadata["targets"]["stress"]["shape"] == (3, 3)
    assert metadata["targets"]["stress"]["extensive"] is False
    assert metadata["targets"]["stress"]["type"] == "per-image"

    # Confirm extensive_property metadata guessed properly
    assert metadata["targets"]["info.extensive_property"]["extensive"] is True
    assert metadata["targets"]["info.extensive_property"]["shape"] == ()
    assert metadata["targets"]["info.extensive_property"]["type"] == "per-image"

    # Confirm tensor_property metadata guessed properly
    assert metadata["targets"]["info.tensor_property"]["extensive"] is False
    assert metadata["targets"]["info.tensor_property"]["shape"] == (6, 6)
    assert metadata["targets"]["info.tensor_property"]["type"] == "per-image"


def test_db_add_delete(tmp_path, structures):
    database = db.connect(tmp_path / "asedb.db")
    for _i, atoms in enumerate(structures):
        database.write(atoms, data=atoms.info)

    dataset = AseDBDataset(config={"src": str(tmp_path / "asedb.db")})
    assert len(dataset) == len(structures)
    orig_len = len(dataset)

    database.delete([1])

    new_structures = [
        build.molecule("CH3COOH", vacuum=4),
        build.bulk("Al"),
    ]

    for _i, atoms in enumerate(new_structures):
        database.write(atoms, data=atoms.info)

    dataset = AseDBDataset(config={"src": str(tmp_path / "asedb.db")})
    assert len(dataset) == orig_len + len(new_structures) - 1


def test_ase_multiread_dataset(tmp_path):
    atoms_objects = [build.bulk("Cu", a=a) for a in np.linspace(3.5, 3.7, 10)]

    energies = np.linspace(1, 0, len(atoms_objects))

    traj = Trajectory(tmp_path / "test.traj", mode="w")

    for atoms, energy in zip(atoms_objects, energies):
        calc = SinglePointCalculator(atoms, energy=energy, forces=atoms.positions)
        atoms.calc = calc
        traj.write(atoms)

    dataset = AseReadMultiStructureDataset(
        config={
            "src": str(tmp_path),
            "pattern": "*.traj",
            "keep_in_memory": True,
            "atoms_transform_args": {
                "skip_always": True,
            },
        }
    )

    assert len(dataset) == len(atoms_objects)

    with open(tmp_path / "test_index_file", "w") as f:
        f.write(f"{tmp_path / 'test.traj'} {len(atoms_objects)}")

    dataset = AseReadMultiStructureDataset(
        config={
            "src": str(tmp_path),
            "index_file": str(tmp_path / "test_index_file"),
        },
    )

    assert len(dataset) == len(atoms_objects)

    dataset = AseReadMultiStructureDataset(
        config={
            "src": str(tmp_path),
            "index_file": str(tmp_path / "test_index_file"),
            "a2g_args": {
                "r_energy": True,
                "r_forces": True,
            },
            "include_relaxed_energy": True,
        }
    )

    assert len(dataset) == len(atoms_objects)

    assert hasattr(dataset[0], "energy_relaxed")
    assert dataset[0].energy_relaxed != dataset[0].energy
    assert dataset[-1].energy_relaxed == dataset[-1].energy

    dataset = AseReadDataset(
        config={
            "src": str(tmp_path),
            "pattern": "*.traj",
            "ase_read_args": {
                "index": "0",
            },
            "a2g_args": {
                "r_energy": True,
                "r_forces": True,
            },
            "include_relaxed_energy": True,
        }
    )

    assert hasattr(dataset[0], "energy_relaxed")
    assert dataset[0].energy_relaxed != dataset[0].energy


def test_empty_dataset(tmp_path):
    # raises error on empty dataset
    with pytest.raises(ValueError):
        AseReadMultiStructureDataset(config={"src": str(tmp_path)})

    with pytest.raises(ValueError):
        AseDBDataset(config={"src": str(tmp_path)})
