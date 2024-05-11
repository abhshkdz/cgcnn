import numpy as np
import pytest

from fairchem.applications.ocpneb.core import OCPNEB
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.optimize import BFGS
from fairchem.core.models.model_registry import model_name_to_local_file

try:
    from ase.mep import DyNEB
except ImportError:  # newest unreleased version has changed imports
    from ase.mep.neb import DyNEB


@pytest.mark.usefixtures("neb_frames")
class TestNEB:
    def test_force_call(self, tmp_path):
        images = self.images
        checkpoint_path = model_name_to_local_file(
            "EquiformerV2-31M-S2EF-OC20-All+MD",
            local_cache=tmp_path / "ocp_checkpoints/",
        )
        batched = OCPNEB(
            images=images,
            checkpoint_path=checkpoint_path,
            cpu=True,
            batch_size=4,
            k=0.25,
        )

        for image in images[1:-1]:
            image.calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)
        unbatched = DyNEB(images, k=0.25)

        forces = batched.get_forces()
        energies = batched.get_potential_energy()

        forces_ub = unbatched.get_forces()
        energies_ub = unbatched.get_potential_energy()

        mismatch = np.isclose(forces, forces_ub, atol=1e-3).all(axis=1) is False

        assert np.isclose(energies, energies_ub, atol=1e-3).all()
        assert mismatch.sum() == 0

    def test_neb_call(self, tmp_path):
        images = self.images.copy()
        checkpoint_path = model_name_to_local_file(
            "EquiformerV2-31M-S2EF-OC20-All+MD",
            local_cache=tmp_path / "ocp_checkpoints/",
        )
        batched = OCPNEB(
            images=images,
            checkpoint_path=checkpoint_path,
            cpu=True,
            batch_size=4,
            k=0.25,
        )

        batched_opt = BFGS(batched, trajectory="batched.traj")
        batched_opt.run(fmax=0.05, steps=5)
        forces = batched.get_forces()
        energies = batched.get_potential_energy()

        images_ub = self.images
        for image in images_ub[1:-1]:
            image.calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)
        unbatched = DyNEB(images_ub, k=0.25)
        unbatched_opt = BFGS(unbatched, trajectory="unbatched.traj")
        unbatched_opt.run(fmax=0.05, steps=5)
        forces_ub = unbatched.get_forces()
        energies_ub = unbatched.get_potential_energy()

        mismatch = np.isclose(forces, forces_ub, atol=1e-3).all(axis=1) is False

        assert np.isclose(energies, energies_ub, atol=1e-3).all()
        assert mismatch.sum() == 0
