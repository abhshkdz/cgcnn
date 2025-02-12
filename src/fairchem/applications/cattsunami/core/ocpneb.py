from __future__ import annotations

import logging

import numpy as np
import torch
from ase.optimize.precon import Precon, PreconImages
from torch.utils.data import DataLoader

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import setup_imports, setup_logging
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing import AtomsToGraphs

try:
    from ase.neb import DyNEB, NEBState
except ImportError:  # newest unreleased version has changed imports
    from ase.mep.neb import DyNEB, NEBState


class OCPNEB(DyNEB):
    def __init__(
        self,
        images,
        checkpoint_path,
        k=0.1,
        fmax=0.05,
        climb=False,
        parallel=False,
        remove_rotation_and_translation=False,
        world=None,
        dynamic_relaxation=True,
        scale_fmax=0.0,
        method="aseneb",
        allow_shared_calculator=False,
        precon=None,
        cpu=False,
        batch_size=4,
        trainer=None,
    ):
        """
        Subclass of NEB that allows for scaled and dynamic optimizations of
        images. This method, which only works in series, does not perform
        force calls on images that are below the convergence criterion.
        The convergence criteria can be scaled with a displacement metric
        to focus the optimization on the saddle point region.

        'Scaled and Dynamic Optimizations of Nudged Elastic Bands',
        P. Lindgren, G. Kastlunger and A. A. Peterson,
        J. Chem. Theory Comput. 15, 11, 5787-5793 (2019).

        dynamic_relaxation: bool
            True skips images with forces below the convergence criterion.
            This is updated after each force call; if a previously converged
            image goes out of tolerance (due to spring adjustments between
            the image and its neighbors), it will be optimized again.
            False reverts to the default NEB implementation.

        fmax: float
            Must be identical to the fmax of the optimizer.

        scale_fmax: float
            Scale convergence criteria along band based on the distance between
            an image and the image with the highest potential energy. This
            keyword determines how rapidly the convergence criteria are scaled.
        """
        super().__init__(
            images,
            k=k,
            climb=climb,
            fmax=fmax,
            dynamic_relaxation=dynamic_relaxation,
            parallel=parallel,
            remove_rotation_and_translation=remove_rotation_and_translation,
            world=world,
            method=method,
            allow_shared_calculator=allow_shared_calculator,
            precon=precon,
            scale_fmax=scale_fmax,
        )
        self.batch_size = batch_size
        setup_imports()
        setup_logging()

        # Silence otf_graph warnings
        logging.disable(logging.WARNING)
        if cpu:
            ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(checkpoint_path)
        config = ckpt["config"]
        if "normalizer" not in config:
            del config["dataset"]["src"]
            config["normalizer"] = config["dataset"]
        if "model_attributes" in config:
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]
        if "relax_dataset" in config["task"]:
            del config["task"]["relax_dataset"]

        if trainer is None:
            trainer = config.get("trainer", "ocp")

        self.trainer = registry.get_trainer_class(trainer)(
            task=config.get("task", {}),
            model=config["model"],
            outputs=config.get("outputs", {}),
            loss_functions=config.get("loss_functions", {}),
            evaluation_metrics=config.get("evaluation_metrics", {}),
            dataset=[config["dataset"]],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=cpu,
            amp=True,
        )

        self.load_checkpoint(checkpoint_path)

        self.a2g = AtomsToGraphs(
            max_neigh=50,
            radius=6,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=False,
            r_pbc=True,
        )

        self.intermediate_energies = []
        self.intermediate_forces = []

        self.cached = False

        # # Handle constraints:
        # fixed_atoms = [idx for idx, tag in enumerate(self.images[0].get_tags()) if tag == 0]
        # fixed_atoms_all = []
        # for image in self.images[1:-1]:
        #     image.constraints = [FixAtoms(indices=fixed_atoms)]

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        try:
            self.trainer.load_checkpoint(checkpoint_path)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def get_forces(self):
        images = self.images[1:-1]
        if self.cached:
            energies = self.intermediate_energies
            forces = self.intermediate_forces
        else:
            energies_calcd = []
            energies = np.empty(self.nimages)
            forces = []
            dataset = self.a2g.convert_all(images, disable_tqdm=True)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=data_list_collater,
                shuffle=False,
                num_workers=2,
            )
            for batch in dataloader:
                predictions = self.trainer.predict(
                    batch, per_image=False, disable_tqdm=True
                )
                energies_calcd.extend(predictions["energy"].flatten().tolist())
                forces.extend(predictions["forces"].cpu().numpy())

            energies[1:-1] = energies_calcd
            forces = np.array(forces)

            # Handle constraints:
            fixed_atoms = np.array(
                [idx for idx, tag in enumerate(self.images[0].get_tags()) if tag == 0]
            )
            for i in range(self.nimages - 2):
                for fixed_atom in fixed_atoms:
                    forces[fixed_atom + len(images[0]) * i] = [0, 0, 0]

            forces = np.reshape(forces, (len(images), self.natoms, 3))
            forces = self.get_precon_forces(forces, energies, self.images)

            self.intermediate_forces = forces
            self.intermediate_energies = energies

        if not self.dynamic_relaxation:
            return forces
        """Get NEB forces and scale the convergence criteria to focus
           optimization on saddle point region. The keyword scale_fmax
           determines the rate of convergence scaling."""
        n = self.natoms
        for i in range(self.nimages - 2):
            n1 = n * i
            n2 = n1 + n
            force = np.sqrt((forces[n1:n2] ** 2.0).sum(axis=1)).max()
            n_imax = (self.imax - 1) * n  # Image with highest energy.

            positions = self.get_positions()
            pos_imax = positions[n_imax : n_imax + n]

            """Scale convergence criteria based on distance between an
               image and the image with the highest potential energy."""
            rel_pos = np.sqrt(((positions[n1:n2] - pos_imax) ** 2).sum())
            if force < self.fmax * (1 + rel_pos * self.scale_fmax):
                if i == self.imax - 1:
                    # Keep forces at saddle point for the log file.
                    pass
                else:
                    # Set forces to zero before they are sent to optimizer.
                    forces[n1:n2, :] = 0
        self.cached = True

        return forces

    def set_positions(self, positions):
        if not self.dynamic_relaxation:
            return super().set_positions(positions)

        n1 = 0
        # old_positions = self.images[0].get_positions()
        # tags_hier = [self.images[i].get_tags() for i in range(self.nimages)]
        # tags = [x for l in tags_hier for x in l]
        for i, image in enumerate(self.images[1:-1]):
            if self.parallel:
                msg = (
                    "Dynamic relaxation does not work efficiently "
                    "when parallelizing over images. Try AutoNEB "
                    "routine for freezing images in parallel."
                )
                raise ValueError(msg)
            else:
                forces_dyn = self._fmax_all(self.images)
                if forces_dyn[i] < self.fmax:
                    n1 += self.natoms
                else:
                    n2 = n1 + self.natoms
                    # new_positions = [old_positions[idx-n1] if tags[idx] == 0 else positions[idx] for idx in range(n1,n2)]
                    # image.set_positions(new_positions)
                    image.set_positions(positions[n1:n2])
                    n1 = n2
        self.cached = False
        return None

    def get_precon_forces(self, forces, energies, images):
        if self.precon is None or isinstance(self.precon, (str, Precon, list)):
            self.precon = PreconImages(self.precon, images)

        # apply preconditioners to transform forces
        # for the default IdentityPrecon this does not change their values
        precon_forces = self.precon.apply(forces, index=slice(1, -1))

        # Save for later use in iterimages:
        self.energies = energies
        self.real_forces = np.zeros((self.nimages, self.natoms, 3))
        self.real_forces[1:-1] = forces

        state = NEBState(self, images, energies)

        # Can we get rid of self.energies, self.imax, self.emax etc.?
        self.imax = state.imax
        self.emax = state.emax

        spring1 = state.spring(0)

        self.residuals = []
        for i in range(1, self.nimages - 1):
            spring2 = state.spring(i)
            tangent = self.neb_method.get_tangent(state, spring1, spring2, i)

            # Get overlap between full PES-derived force and tangent
            tangential_force = np.vdot(forces[i - 1], tangent)

            # from now on we use the preconditioned forces (equal for precon=ID)
            imgforce = precon_forces[i - 1]

            if i == self.imax and self.climb:
                """The climbing image, imax, is not affected by the spring
                forces. This image feels the full PES-derived force,
                but the tangential component is inverted:
                see Eq. 5 in paper II."""
                if self.method == "aseneb":
                    tangent_mag = np.vdot(tangent, tangent)  # For normalizing
                    imgforce -= 2 * tangential_force / tangent_mag * tangent
                else:
                    imgforce -= 2 * tangential_force * tangent
            else:
                self.neb_method.add_image_force(
                    state, tangential_force, tangent, imgforce, spring1, spring2, i
                )
                # compute the residual - with ID precon, this is just max force
                residual = self.precon.get_residual(i, imgforce)
                self.residuals.append(residual)

            spring1 = spring2

        return precon_forces.reshape((-1, 3))
