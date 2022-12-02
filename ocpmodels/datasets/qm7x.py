import random
import re
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import spatial as sp
from torch import as_tensor
from torch_geometric.data import Data

from cosmosis.dataset import CDataset


class Molecule:
    """an abstract class with utilities for creating molecule instances"""

    lookup = {
        "bond_type": {"misc": 1, "SINGLE": 2, "DOUBLE": 3, "TRIPLE": 4, "AROMATIC": 5},
        "stereo": {
            "STEREONONE": 1,
            "STEREOZ": 2,
            "STEREOE": 3,
            "STEREOCIS": 4,
            "STEREOTRANS": 5,
            "STEREOANY": 6,
        },
        "atomic_numbers": {"C": 6, "H": 1, "N": 7, "O": 8, "F": 9},
        "hybridization": {
            "UNSPECIFIED": 1,
            "S": 2,
            "SP": 3,
            "SP2": 4,
            "SP3": 5,
            "SP3D": 6,
            "SP3D2": 7,
            "OTHER": 8,
        },
        "chirality": {
            "CHI_UNSPECIFIED": 1,
            "CHI_TETRAHEDRAL_CW": 2,
            "CHI_TETRAHEDRAL_CCW": 3,
            "CHI_OTHER": 4,
        },
    }

    rdkit_features = [
        "atom_types",
        "atomic_numbers",
        "aromatic",
        "chirality",
        "degree",
        "charge",
        "n_hs",
        "n_rads",
        "hybridization",
        "edge_indices",
        "edge_attr",
        "rdmol_block",
        "n_atoms",
        "xyz",
        "distance",
        "coulomb",
        "adjacency",
        "rdmol",
    ]

    @abstractmethod
    def __repr__(self):
        return self.mol_id

    @abstractmethod
    def load_molecule(self, *args):
        self.smile
        self.rdmol
        self.mol_block
        self.xyz
        self.distance
        self.atom_types
        self.n_atoms
        self.atomic_numbers

    def open_file(self, in_file):
        with open(in_file) as f:
            data = []
            for line in f.readlines():
                data.append(line)
            return data

    def rdmol_from_smile(self, smile):
        self.rdmol = Chem.AddHs(Chem.MolFromSmiles(smile))

    def adjacency_from_rdmol_block(self, rdmol_block):
        """use the V2000 chemical table's (rdmol MolBlock) adjacency list to create a
        nxn symetric matrix with 0, 1, 2 or 3 for bond type where n is the indexed
        atom"""
        adjacency = np.zeros((self.n_atoms, self.n_atoms), dtype="int64")
        block = rdmol_block.split("\n")
        for b in block[:-2]:
            line = "".join(b.split())
            if len(line) == 4:
                # shift -1 to index from zero
                adjacency[(int(line[0]) - 1), (int(line[1]) - 1)] = int(line[2])
                # create bi-directional connection
                adjacency[(int(line[1]) - 1), (int(line[0]) - 1)] = int(line[2])
        return adjacency

    def embed_rdmol(self, rdmol, n_conformers):
        AllChem.EmbedMultipleConfs(
            rdmol,
            numConfs=n_conformers,
            maxAttempts=0,
            useRandomCoords=False,
            numThreads=0,
        )

    def create_rdmol_data(self, rdmol):
        atom_types = []
        atomic_numbers = []
        aromatic = []
        chirality = []
        degree = []
        charge = []
        n_hs = []
        n_rads = []
        hybridization = []

        for atom in rdmol.GetAtoms():
            atom_types.append(atom.GetSymbol())
            atomic_numbers.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            chirality.append(str(atom.GetChiralTag()))
            degree.append(atom.GetTotalDegree())
            charge.append(atom.GetFormalCharge())
            n_hs.append(atom.GetTotalNumHs())
            n_rads.append(atom.GetNumRadicalElectrons())
            hybridization.append(str(atom.GetHybridization()))

        self.atom_types = np.asarray(atom_types)
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=np.float32)
        self.aromatic = np.asarray(aromatic, dtype=np.float32)
        self.chirality = np.asarray(chirality)
        self.degree = np.asarray(degree, dtype=np.float32)
        self.charge = np.asarray(charge, dtype=np.float32)
        self.n_hs = np.asarray(n_hs, dtype=np.float32)
        self.n_rads = np.asarray(n_rads, dtype=np.float32)
        self.hybridization = np.asarray(hybridization)

        for bond in rdmol.GetBonds():
            edge_indices, edge_attrs = [], []
            for bond in rdmol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                e.append(Molecule.lookup["bond_type"][str(bond.GetBondType())])
                e.append(Molecule.lookup["stereo"][str(bond.GetStereo())])
                e.append(1 if bond.GetIsConjugated() else 0)
                e.append(1 if atom.IsInRing() else 0)

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

        self.edge_indices = np.reshape(
            np.asarray(edge_indices, dtype=np.int64), (-1, 2)
        )
        self.edge_attr = np.reshape(np.asarray(edge_attrs, dtype=np.float32), (-1, 4))

        self.rdmol_block = Chem.MolToMolBlock(rdmol)
        self.n_atoms = int(rdmol.GetNumAtoms())

    def xyz_from_rdmol(self, rdmol):
        """
        TODO:
        load all the conformer xyz and choose among them at runtime (_get_features())
        """
        if rdmol.GetNumConformers() == 0:
            return []
        else:
            confs = self.rdmol.GetConformers()
            conf = random.choice(confs)
            xyz = conf.GetPositions()
        return xyz

    def distance_from_xyz(self, xyz):
        m = np.zeros((len(xyz), 3))
        for i, atom in enumerate(xyz):
            m[i, :] = atom
        distance = sp.distance.squareform(sp.distance.pdist(m)).astype("float32")
        return distance

    def create_coulomb(self, distance, atomic_numbers, sigma=1):
        """creates coulomb matrix obj attr.  set sigma to False to turn off random sorting.
        sigma = stddev of gaussian noise.
        https://papers.nips.cc/paper/4830-learning-invariant-representations-of-\
        molecules-for-atomization-energy-prediction"""

        qmat = atomic_numbers[None, :] * atomic_numbers[:, None]
        idmat = np.linalg.inv(distance)
        np.fill_diagonal(idmat, 0)
        coul = qmat @ idmat
        np.fill_diagonal(coul, 0.5 * atomic_numbers**2.4)
        if sigma:
            coulomb = self.sort_permute(coul, sigma)
        else:
            coulomb = coul
        return coulomb

    def sort_permute(self, matrix, sigma):
        norm = np.linalg.norm(matrix, axis=1)
        noised = np.random.normal(norm, sigma)
        indexlist = np.argsort(noised)
        indexlist = indexlist[::-1]  # invert
        return matrix[indexlist][:, indexlist]


class BaseQM7X(Molecule, CDataset):
    # code from https://github.com/icanswim/qchem/blob/main/dataset.py#L640
    """QM7-X: A comprehensive dataset of quantum-mechanical properties spanning
    the chemical space of small organic molecules
    https://arxiv.org/abs/2006.15139

    Dataset source/download:
    https://zenodo.org/record/3905361

    Decompress the .xz files in the 'in_dir' folder (qchem/data/qm7x/)

    1000.hdf5 6.5 GB
    2000.hdf5 8.8 GB
    3000.hdf5 16.9 GB
    4000.hdf5 12.4 GB
    5000.hdf5 9.8 GB
    6000.hdf5 17.2 GB
    7000.hdf5 9.8 GB
    8000.hdf5 0.8 GB

    A description of the structure generation procedure is available in the paper
    related to this dataset.  Each HDF5 file contains information about the molecular
    properties of equilibrium and non-equilibrium conformations of small molecules
    composed of up to seven heavy atoms (C, N, O, S, Cl). For instance, you can access
    to the information saved in the 1000.hdf5 file as,
    fDFT = h5py.File('1000.hdf5', 'r')
    fDFT[idmol]: idmol, ID number of molecule (e.g., '1', '100', '94')
    fDFT[idmol][idconf]: idconf, ID configuration (e.g., 'Geom-m1-i1-c1-opt', 'Geom-m1-i1-c1-50')
    The idconf label has the general form "Geom-mr-is-ct-u", were r enumerated the
    SMILES strings, s the stereoisomers excluding conformers, t the considered
    (meta)stable conformers, and u the optimized/displaced structures; u = opt
    indicates the DFTB3+MBD optimized structures and u = 1,...,100 enumerates
    the displaced non-equilibrium structures. Note that these indices are not
    sorted according to their PBE0+MBD relative energies.
    Then, for each structure (i.e., idconf), you will find the following properties:
    -'atNUM': Atomic numbers (N)
    -'atXYZ': Atoms coordinates [Ang] (Nx3)
    -'sRMSD': RMSD to optimized structure [Ang] (1)
    -'sMIT': Momente of inertia tensor [amu.Ang^2] (9)
    -'ePBE0+MBD': Total PBE0+MBD energy [eV] (1)
    -'eDFTB+MBD': Total DFTB+MBD energy [eV] (1)
    -'eAT': PBE0 atomization energy [eV] (1)
    -'ePBE0': PBE0 energy [eV] (1)
    -'eMBD': MBD energy [eV] (1)
    -'eTS': TS dispersion energy [eV] (1)
    -'eNN': Nuclear-nuclear repulsion energy [eV] (1)
    -'eKIN': Kinetic energy [eV] (1)
    -'eNE': Nuclear-electron attracttion [eV] (1)
    -'eEE': Classical coulomb energy (el-el) [eV] (1)
    -'eXC': Exchange-correlation energy [eV] (1)
    -'eX': Exchange energy [eV] (1)
    -'eC': Correlation energy [eV] (1)
    -'eXX': Exact exchange energy [eV] (1)
    -'eKSE': Sum of Kohn-Sham eigenvalues [eV] (1)
    -'KSE': Kohn-Sham eigenvalues [eV] (depends on the molecule)
    -'eH': HOMO energy [eV] (1)
    -'eL': LUMO energy [eV] (1)
    -'HLgap': HOMO-LUMO gap [eV] (1)
    -'DIP': Total dipole moment [e.Ang] (1)
    -'vDIP': Dipole moment components [e.Ang] (3)
    -'vTQ': Total quadrupole moment components [e.Ang^2] (3)
    -'vIQ': Ionic quadrupole moment components [e.Ang^2] (3)
    -'vEQ': Electronic quadrupole moment components [eAng^2] (3)
    -'mC6': Molecular C6 coefficient [hartree.bohr^6] (computed using SCS) (1)
    -'mPOL': Molecular polarizability [bohr^3] (computed using SCS) (1)
    -'mTPOL': Molecular polarizability tensor [bohr^3] (9)
    -'totFOR': Total PBE0+MBD atomic forces (unitary forces cleaned) [eV/Ang] (Nx3)
    -'vdwFOR': MBD atomic forces [eV/Ang] (Nx3)
    -'pbe0FOR': PBE0 atomic forces [eV/Ang] (Nx3)
    -'hVOL': Hirshfeld volumes [bohr^3] (N)
    -'hRAT': Hirshfeld ratios (N)
    -'hCHG': Hirshfeld charges [e] (N)
    -'hDIP': Hirshfeld dipole moments [e.bohr] (N)
    -'hVDIP': Components of Hirshfeld dipole moments [e.bohr] (Nx3)
    -'atC6': Atomic C6 coefficients [hartree.bohr^6] (N)
    -'atPOL': Atomic polarizabilities [bohr^3] (N)
    -'vdwR': van der Waals radii [bohr] (N)

    'distance': N x N distance matrix created from atXYZ

    selector = list of regular expression strings (attr) for searching
        and selecting idconf keys.
        idconf = ID configuration (e.g., 'Geom-m1-i1-c1-opt', 'Geom-m1-i1-c1-50')
    flatten = True/False
    pad = None/int (pad length int in the Na (number of atoms) dimension)

    returns datadic[idmol][idconf][properties]
    """

    set_ids = ["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000"]

    properties = [
        "DIP",
        "HLgap",
        "KSE",
        "atC6",
        "atNUM",
        "atPOL",
        "atXYZ",
        "eAT",
        "eC",
        "eDFTB+MBD",
        "eEE",
        "eH",
        "eKIN",
        "eKSE",
        "eL",
        "eMBD",
        "eNE",
        "eNN",
        "ePBE0",
        "ePBE0+MBD",
        "eTS",
        "eX",
        "eXC",
        "eXX",
        "hCHG",
        "hDIP",
        "hRAT",
        "hVDIP",
        "hVOL",
        "mC6",
        "mPOL",
        "mTPOL",
        "pbe0FOR",
        "sMIT",
        "sRMSD",
        "totFOR",
        "vDIP",
        "vEQ",
        "vIQ",
        "vTQ",
        "vdwFOR",
        "vdwR",
        "distance",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, i):
        # if multiple conformations one is randomly selected
        conformations = list(self.ds[i].keys())
        idconf = random.choice(conformations)

        datadic = {}
        if len(self.features) > 0:
            X = self._get_features(self.ds[i][idconf], self.features)
            for transform in self.transform:
                X = transform(X)
            if self.as_tensor:
                X = as_tensor(X)
            datadic["X"] = X

        if len(self.embeds) > 0:
            embed_idx = self._get_embed_idx(
                self.ds[i][idconf], self.embeds, self.embed_lookup
            )
            datadic["embed_idx"] = embed_idx

        if len(self.targets) > 0:
            y = self._get_features(self.ds[i][idconf], self.targets)
            for transform in self.target_transform:
                y = transform(y)
            if self.as_tensor:
                X = as_tensor(X)
            datadic["y"] = y

        return datadic

    def _get_features(self, datadic, features):
        data = {}
        for f in features:
            out = datadic[f]

            if self.pad is not None:
                # (Nc, Na), (Nc, Na, Na)
                if f in ["atNUM", "distance", "coulomb"]:
                    out = np.pad(out, ((0, (self.pad - out.shape[0]))))
                # (Nc, Na, 3), (Nc, Na, 1)
                elif f in [
                    "atXYZ",
                    "totFOR",
                    "vdwFOR",
                    "pbe0FOR",
                    "hVDIP",
                    "atC6",
                    "hVOL",
                    "hRAT",
                    "hCHG",
                    "atC6",
                    "atPOL",
                    "vdwR",
                    "hDIP",
                ]:
                    out = np.pad(out, ((0, (self.pad - out.shape[0])), (0, 0)))
                # (Nc, 9), (Nc, 3), (Nc) no padding
            if self.flatten:
                out = np.reshape(out, -1)

            data[f] = out

        return data

    def _load_features(self, mol, features, dtype="float32"):
        datadic = {}
        for f in features:
            if f == "distance":
                out = mol["atXYZ"][()]
                out = sp.distance.squareform(sp.distance.pdist(out))
            elif f == "coulomb":
                out = mol["atXYZ"][()]
                distance = sp.distance.squareform(sp.distance.pdist(out))
                atomic_numbers = mol["atNUM"][()]
                out = self.create_coulomb(distance, atomic_numbers)
            else:
                out = mol[f][()]
            datadic[f] = out.astype(dtype)
        return datadic

    def load_data(self, selector="opt", in_dir="./data/qm7x/"):
        """selector = list of regular expression strings (attr) for searching
        and selecting idconf keys.
        returns datadic[idmol] = {'idconf': {'feature': val}}
        idconf = ID configuration (e.g., 'Geom-m1-i1-c1-opt', 'Geom-m1-i1-c1-50')
        datadic[idmol][idconf][feature]
        """
        datadic = {}
        structure_count = 0
        for set_id in self.set_ids:
            with h5py.File(in_dir + set_id + ".hdf5", "r") as f:
                for idmol in f:
                    print(
                        set_id,
                        str(idmol).zfill(4),
                        str(structure_count).zfill(5),
                        end="\r",
                        flush=True,
                    )
                    datadic[int(idmol)] = {}
                    for idconf in f[idmol]:
                        for attr in selector:
                            if re.search(attr, idconf):
                                structure_count += 1
                                features = self._load_features(
                                    f[idmol][idconf],
                                    self.features + self.targets + self.embeds,
                                )
                                datadic[int(idmol)][idconf] = features
                            if hasattr(self, "max_structures"):
                                if (
                                    self.max_structures > 0
                                    and structure_count >= self.max_structures
                                ):
                                    print(
                                        "molecular formula (idmol) mapped: ",
                                        len(datadic),
                                    )
                                    print(
                                        "total molecular structures (idconf) mapped: ",
                                        structure_count,
                                    )
                                    return datadic

        print("molecular formula (idmol) mapped: ", len(datadic))
        print("total molecular structures (idconf) mapped: ", structure_count)
        return datadic


class InMemoryQM7X(BaseQM7X):
    def __init__(
        self,
        in_dir,
        features=["atXYZ", "atNUM"],
        ignore_features=["atPOL", "atC6"],
        y="eX",
        attribute_remapping={"atNUM": "z", "atXYZ": "pos"},
        max_structures=-1,
        selector=[".+"],
        set_ids=None,
    ):
        ignores = set(ignore_features)
        self.attribute_remapping = attribute_remapping
        self.y = y
        self.max_structures = max_structures

        if set_ids is not None:
            assert isinstance(set_ids, list)
            assert all(s in BaseQM7X.set_ids for s in set_ids)
            self.set_ids = set_ids

        super().__init__(
            in_dir=in_dir,
            features=features,
            targets=sorted(set(BaseQM7X.properties) - ignores - set(features)),
            selector=selector,
        )

        self.sample_mapping = []
        if self.ds:
            for idmol in self.ds:
                for idconf in self.ds[idmol]:
                    self.sample_mapping.append((idmol, idconf))

    def __len__(self):
        return len(self.sample_mapping)

    def from_ds(self, idmol, idconf):
        return self.ds[idmol][idconf]

    def __getitem__(self, i):
        idmol, idconf = self.sample_mapping[i]

        conformation = self.from_ds(idmol, idconf)

        datadic = {}
        if len(self.features) > 0:
            x_dic = self._get_features(conformation, self.features)
            datadic.update(x_dic)

        if len(self.embeds) > 0:
            embed_idx = self._get_embed_idx(
                conformation, self.embeds, self.embed_lookup
            )
            datadic["embed_idx"] = embed_idx

        if len(self.targets) > 0:
            y_dic = self._get_features(conformation, self.targets)
            datadic.update(y_dic)

        # remap attributes
        for k, v in self.attribute_remapping.items():
            datadic[v] = datadic.pop(k)

        if self.y:
            # set y attribute
            assert self.y in datadic, f"y ({self.y}) not found in data dict " + str(
                datadic
            )
            datadic["y"] = datadic.pop(self.y)

        data = Data(**datadic)

        if self.transform is not None:
            if callable(self.transform):
                data = self.transform(data)
            elif isinstance(self.transform, Iterable):
                for transform in self.transform:
                    data = transform(data)
            else:
                raise ValueError(
                    "`transform` must be None, callable or iterable. Received: "
                    + str(self.transform)
                )

        data.idmol = idmol
        data.idconf = idconf

        return data


class QM7X(InMemoryQM7X):
    def __init__(
        self,
        in_dir,
        features=["atXYZ", "atNUM"],
        ignore_features=["atPOL", "atC6"],
        y="eX",
        attribute_remapping={"atNUM": "z", "atXYZ": "pos"},
        max_structures=-1,
        selector=[".+"],
    ):
        self.attribute_remapping = attribute_remapping
        self.y = y
        self.max_structures = max_structures
        super().__init__(
            in_dir,
            features=features,
            ignore_features=ignore_features,
            y=y,
            attribute_remapping=attribute_remapping,
            max_structures=max_structures,
            selector=selector,
        )

        self.sample_mapping = []
        for idmol in self.ds:
            for d, data in enumerate(torch.load(self.ds[idmol])):
                assert (
                    data.idmol == idmol
                ), f"idmol mismatch: data ({data.idmol}) != ds index ({idmol})"
                self.sample_mapping.append((idmol, d))

    def load_data(self, selector, in_dir):
        paths = Path(in_dir).glob("*.pt")
        datadict = defaultdict(dict)
        for p in paths:
            idmol = int(p.stem)
            datadict[idmol] = str(p.resolve())
        return datadict

    def from_ds(self, idmol, idconf):
        return torch.load(self.ds[idmol])[idconf]
