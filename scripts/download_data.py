import argparse
import glob
import os

import ocpmodels

"""
This script provides users with an automated way to download, preprocess (where
applicable), and organize data to readily be used by the existing config files.
"""

DOWNLOAD_LINKS = {
    "s2ef": {
        "200k": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar",
        "2M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar",
        "20M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_20M.tar",
        "all": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_all.tar",
        "val_id": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar",
        "val_ood_ads": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_ads.tar",
        "val_ood_cat": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_cat.tar",
        "val_ood_both": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar",
        "test": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_test_lmdbs.tar.gz",
    },
    "is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz",
}

S2EF_COUNTS = {
    "s2ef": {
        "200k": 200000,
        "2M": 2000000,
        "20M": 20000000,
        "all": 133934018,
        "val_id": 999866,
        "val_ood_ads": 999838,
        "val_ood_cat": 999809,
        "val_ood_both": 999944,
    },
}


def get_data(task, split, del_intmd_files, path):
    if path:
        datadir = os.path.join(path, "data")
    else:
        datadir = os.path.join(os.path.dirname(ocpmodels.__path__[0]), "data")
    os.makedirs(datadir, exist_ok=True)

    if task == "s2ef" and split is None:
        raise NotImplementedError("S2EF requires a split to be defined.")

    if task == "s2ef":
        assert (
            split in DOWNLOAD_LINKS[task]
        ), f'S2EF "{split}" split not defined, please specify one of the following: {list(DOWNLOAD_LINKS["s2ef"].keys())}'
        download_link = DOWNLOAD_LINKS[task][split]

    elif task == "is2re":
        download_link = DOWNLOAD_LINKS[task]

    os.system(f"wget {download_link}")
    filename = os.path.basename(download_link)
    print("Extracting contents...")
    os.system(f"tar -xf {filename}")
    dirname = filename.split(".")[0]
    if task == "s2ef" and split != "test":
        compressed_dir = os.path.join(dirname, dirname)
        if split in ["200k", "2M", "20M", "all"]:
            output_path = os.path.join(datadir, task, split, "train")
        else:
            output_path = os.path.join(datadir, task, "all", split)
        uncompressed_dir = uncompress_data(compressed_dir)
        preprocess_data(uncompressed_dir, output_path)

        verify_count(output_path, task, split)
    if task == "s2ef" and split == "test":
        os.system(f"mv {dirname}/test_data/s2ef/all/test_* {datadir}/s2ef/all")
    elif task == "is2re":
        os.system(f"mv {dirname}/data/is2re {datadir}")

    if del_intmd_files:
        cleanup(filename, dirname)


def uncompress_data(compressed_dir):
    import uncompress

    parser = uncompress.get_parser()
    args, _ = parser.parse_known_args()
    args.ipdir = compressed_dir
    args.opdir = os.path.dirname(compressed_dir) + "_uncompressed"
    uncompress.main(args)
    return args.opdir


def preprocess_data(uncompressed_dir, output_path):
    import preprocess_ef as preprocess

    parser = preprocess.get_parser()
    args, _ = parser.parse_known_args()
    args.data_path = uncompressed_dir
    args.out_path = output_path
    preprocess.main(args)


def verify_count(output_path, task, split):
    paths = glob.glob(os.path.join(output_path, "*.txt"))
    count = 0
    for path in paths:
        lines = open(path, "r").read().splitlines()
        count += len(lines)
    assert (
        count == S2EF_COUNTS[task][split]
    ), f"S2EF {split} count incorrect, verify preprocessing has completed successfully."


def cleanup(filename, dirname):
    import shutil

    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    if os.path.exists(dirname + "_uncompressed"):
        shutil.rmtree(dirname + "_uncompressed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task to download")
    parser.add_argument(
        "--split", type=str, help="Corresponding data split to download"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep intermediate directories and files upon data retrieval/processing",
    )
    # Flags for S2EF train/val set preprocessing:
    parser.add_argument(
        "--get-edges",
        action="store_true",
        help="Store edge indices in LMDB, ~10x storage requirement. Default: compute edge indices on-the-fly.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    parser.add_argument(
        "--ref-energy", action="store_true", help="Subtract reference energies"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Specify the path for ocp directory (in case module is not installed in same location)",
    )

    args, _ = parser.parse_known_args()
    get_data(task=args.task, split=args.split, del_intmd_files=not args.keep, path=args.path)
