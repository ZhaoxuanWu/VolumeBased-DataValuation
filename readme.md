# Validation Free and Replication Robust Volume-based Data Valuation [NeurIPS'2021]

This repository is the official implementation of the following paper accepted by the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS) 2021:

> Xinyi Xu*, Zhaoxuan Wu*, Chuan Sheng Foo, Bryan Kian Hsiang Low
>
> Validation Free and Replication Robust Volume-based Data Valuation [paper](https://proceedings.neurips.cc/paper/2021/hash/59a3adea76fadcb6dd9e54c96fc155d1-Abstract.html)

## Requirements

To install requirements:
```setup
conda env create -f environment.yml
```

## Run synthetic data on baseline distributions
First modify the 'CONFIGS' section in the `main.py` code, then
```bash
mkdir outputs
python main.py
```

## Use real-world datasets
Follow the instructions in the [data/](data/) folder.

In the `readme.md` file under each dataset directory, we specify the collection of dataset files to download and put under the directory.

## Plotting the results
Most of the code for plotting figures can be found in the following jupyter notebooks:
- [notebooks/Experiments_plots.ipynb](notebooks/Experiments_plots.ipynb)
- [notebooks/volume_numerical_experiments.ipynb](notebooks/volume_numerical_experiments.ipynb)

## Other experiments
Code for all other experiments used in the paper can be found unber the following directories:
- [robustness experiments](robustness%20experiments/)
- [other experiments](other%20experiments/)

## Citing
If you have found our work to be useful in your research, please consider citing it with the following bibtex:
```
@inproceedings{Xu2021,
 author = {Xu, Xinyi and Wu, Zhaoxuan and Foo, Chuan Sheng and Low, Bryan Kian Hsiang},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {10837--10848},
 publisher = {Curran Associates, Inc.},
 title = {Validation Free and Replication Robust Volume-based Data Valuation},
 volume = {34},
 year = {2021}
}

```
