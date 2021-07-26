# Open Catalyst Project models

[![CircleCI](https://circleci.com/gh/Open-Catalyst-Project/ocp.svg?style=shield)](https://circleci.com/gh/Open-Catalyst-Project/ocp)

ocp-models is the modeling codebase for the [Open Catalyst Project](https://opencatalystproject.org/).

It provides implementations of state-of-the-art ML algorithms for catalysis that
take arbitrary chemical structures as input to predict energy / forces / positions:

- [DimeNet++](https://arxiv.org/abs/2011.14115)
- [ForceNet](https://arxiv.org/abs/2103.01436)
- [DimeNet](https://arxiv.org/abs/2003.03123)
- [SchNet](https://arxiv.org/abs/1706.08566)
- [CGCNN](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)
- [GROVER](https://arxiv.org/abs/2007.02835)

## Installation

The easiest way to install prerequisites is via [conda](https://conda.io/docs/index.html).

After installing [conda](http://conda.pydata.org/), run the following commands
to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
named `ocp-models` and install dependencies.

### Pre-install step

Install `conda-merge`:
```bash
pip install conda-merge
```
If you're using system `pip`, then you may want to add the `--user` flag to avoid using `sudo`.
Check that you can invoke `conda-merge` by running `conda-merge -h`.

### GPU machines

Instructions are for PyTorch 1.8.1, CUDA 10.2 specifically.

First, check that CUDA is in your `PATH` and `LD_LIBRARY_PATH`, e.g.
```bash
$ echo $PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/10.2/bin

$ echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/10.2/lib64
```

The exact paths may differ on your system.

Then install the dependencies:
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

## Download data

Dataset download links for all tasks can be found at [DATASET.md](https://github.com/Open-Catalyst-Project/ocp/blob/master/DATASET.md).

IS2* datasets are stored as LMDB files and are ready to be used upon download.
S2EF train+val datasets require an additional preprocessing step.

For convenience, a self-contained script can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/scripts/download_data.py) to download, preprocess, and organize the data directories to be readily usable by the existing [configs](https://github.com/Open-Catalyst-Project/ocp/tree/master/configs).

For IS2*, run the script as:

```bash
python scripts/download_data.py --task is2re
```

For S2EF train/val, run the script as:

```bash
python scripts/download_data.py --task s2ef --split SPLIT_SIZE --get-edges --num-workers WORKERS --ref-energy
```

- `--split`: split size to download: `"200k", "2M", "20M", "all", "val_id", "val_ood_ads", "val_ood_cat", or "val_ood_both"`.
- `--get-edges`: includes edge information in LMDBs (~10x storage requirement, ~3-5x slowdown), otherwise, compute edges on the fly (larger GPU memory requirement).
- `--num-workers`: number of workers to parallelize preprocessing across.
- `--ref-energy`: uses referenced energies instead of raw energies.

For S2EF test, run the script as:
```bash
python scripts/download_data.py --task s2ef --split test
```


## Train and evaluate models

A detailed description of how to train and evaluate models, run ML-based
relaxations, and generate EvalAI submission files can be found
[here](https://github.com/Open-Catalyst-Project/ocp/blob/master/TRAIN.md).

Our evaluation server is [hosted on EvalAI](https://eval.ai/web/challenges/challenge-page/712/overview).
Numbers (in papers, etc.) should be reported from the evaluation server.

## Pretrained models

Pretrained model weights accompanying [our paper](https://arxiv.org/abs/2010.09990) are available [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/MODELS.md).

### Tutorials

Interactive tutorial notebooks can be found [here](https://github.com/Open-Catalyst-Project/ocp/tree/master/docs/source/tutorials) to help familirize oneself with various components of the repo:

- [Data visualization](https://github.com/Open-Catalyst-Project/ocp/blob/tutorials/docs/source/tutorials/data_visualization.ipynb) - understanding the raw data and its contents.
- [Data preprocessing](https://github.com/Open-Catalyst-Project/ocp/blob/tutorials/docs/source/tutorials/data_preprocessing.ipynb) - preprocessing raw ASE atoms objects to OCP graph Data objects.
- [LMDB dataset creation](https://github.com/Open-Catalyst-Project/ocp/blob/tutorials/docs/source/tutorials/lmdb_dataset_creation.ipynb) - creating your own OCP-compatible LMDB datasets from ASE-compatible Atoms objects.
- [S2EF training example](https://github.com/Open-Catalyst-Project/ocp/blob/tutorials/docs/source/tutorials/train_s2ef_example.ipynb) - training a SchNet S2EF model, loading a trained model, and making predictions.

## Discussion

For all non-codebase related questions and to keep up-to-date with the latest OCP announcements, please join the [discussion board](https://discuss.opencatalystproject.org/). All codebase related questions and issues should be posted directly on our [issues page](https://github.com/Open-Catalyst-Project/ocp/issues).

## Acknowledgements

- This codebase was initially forked from [CGCNN](https://github.com/txie-93/cgcnn)
by [Tian Xie](http://txie.me), but has undergone significant changes since.
- A lot of engineering ideas have been borrowed from [github.com/facebookresearch/mmf](https://github.com/facebookresearch/mmf).
- The DimeNet++ implementation is based on the [author's Tensorflow implementation](https://github.com/klicperajo/dimenet) and the [DimeNet implementation in Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py).

## Citation

If you use this codebase in your work, consider citing:

```bibtex
@article{ocp_dataset,
    author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
    title = {Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    journal = {ACS Catalysis},
    year = {2021},
    doi = {10.1021/acscatal.0c04525},
}
```

## License

[MIT](https://github.com/Open-Catalyst-Project/ocp/blob/master/LICENSE.md)
