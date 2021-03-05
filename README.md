# Open-Catalyst-Project Models

Implements the following baselines that take arbitrary chemical structures as
input to predict material properties:
- [DimeNet++](https://arxiv.org/abs/2011.14115)
- [DimeNet](https://arxiv.org/abs/2003.03123)
- [SchNet](https://arxiv.org/abs/1706.08566)
- [CGCNN](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301).

##  Installation

[last updated December 09, 2020]

The easiest way of installing prerequisites is via [conda](https://conda.io/docs/index.html).
After installing [conda](http://conda.pydata.org/), run the following commands
to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
named `ocp-models` and install dependencies:

### Pre-install step

Install `conda-merge`:
```bash
pip install conda-merge
```
If you're using system `pip`, then you may want to add the `--user` flag to avoid using `sudo`.
Check that you can invoke `conda-merge` by running `conda-merge -h`.

### GPU machines

Instructions are for PyTorch 1.6, CUDA 10.1 specifically.

First, check that CUDA is in your `PATH` and `LD_LIBRARY_PATH`, e.g.
```
$echo $PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/10.1/bin
$echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/10.1/lib64
```
The exact paths may differ on your system. Then install the dependencies:
```bash
conda-merge env.common.yml env.gpu.yml > env.yml
conda env create -f env.yml
```
Activate the conda environment with `conda activate ocp-models`.
Install this package with `pip install -e .`.
Finally, install the pre-commit hooks:
```bash
pre-commit install
```

### CPU-only machines

Please skip the following if you completed the with-GPU installation from above.

```bash
conda-merge env.common.yml env.cpu.yml > env.yml
conda env create -f env.yml
conda activate ocp-models
pip install -e .
pre-commit install
```

### Google Colab Notebooks

[Google Colab](https://colab.research.google.com/) provides free access to GPU-accelerated Jupyter notebooks—train a notebook right from your browser!
- S2EF training example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elliottower/ocp/blob/696DS-Spring21/docs/source/tutorials/colab_train_s2ef_example.ipynb
)

## Usage

### Project website

The project website is [opencatalystproject.org](https://opencatalystproject.org). Links to dataset paper and the whitepaper can be found on the website.

### Download the datasets

Dataset download links can be found at [DATASET.md](https://github.com/Open-Catalyst-Project/ocp/blob/master/DATASET.md) for the S2EF, IS2RS, and IS2RE tasks. IS2* datasets are stored as LMDB files and are ready to be used upon download. S2EF train+val datasets require an additional preprocessing step. For convenience, a self-contained script can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/scripts/download_data.py) to download, preprocess, and organize the data directories to be readily usable by the existing [configs](https://github.com/Open-Catalyst-Project/ocp/tree/master/configs):

IS2* datasets: `python scripts/download_data.py --task is2re`

S2EF datasets:
- train/val splits: `python scripts/download_data.py --task s2ef --split SPLIT_SIZE --get-edges --num-workers WORKERS --ref-energy`; where
    - `--get-edges`: includes edge information in LMDBs (~10x storage requirement, ~3-5x slowdown), otherwise, compute edges on the fly (larger GPU memory requirement).
    - `--ref-energy`: uses referenced energies instead of raw energies.
    - `--split`: split size to download: `"200k", "2M", "20M", "all", "val_id", "val_ood_ads", "val_ood_cat", or "val_ood_both"`.
    - `--num-workers`: number of workers to parallelize preprocessing across.
- test splits: `python scripts/download_data.py --task s2ef --split test`

An interactive notebook can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/docs/source/tutorials/data_playground.ipynb) to provide some intution on the data and its contents.

### Train models for the desired tasks

A detailed description of how to train, predict, and run ML-based relaxations can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/TRAIN.md).

A simplified interactive notebook example can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/docs/source/tutorials/train_s2ef_example.ipynb).

### Pretrained models

Pretrained models accompanying https://arxiv.org/abs/2010.09990v2 can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/MODELS.md).

## Discussions/FAQs

For all non-codebase related questions and to keep up-to-date with the latest OCP announcements, please join the [discussion board](https://discuss.opencatalystproject.org/). All codebase related questions and issues should be posted directly on our [issues page](https://github.com/Open-Catalyst-Project/ocp/issues).

## Acknowledgements

- This codebase was initially forked from [CGCNN](https://github.com/txie-93/cgcnn)
by [Tian Xie](http://txie.me), but has undergone significant changes since.
- A lot of engineering ideas have been borrowed from [github.com/facebookresearch/mmf](https://github.com/facebookresearch/mmf).
- The DimeNet++ implementation is based on the [author's Tensorflow implementation](https://github.com/klicperajo/dimenet) and the [DimeNet implementation in Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py).

## License

This code is MIT licensed, as found in the [LICENSE file](https://github.com/Open-Catalyst-Project/ocp/blob/master/LICENSE.md).
