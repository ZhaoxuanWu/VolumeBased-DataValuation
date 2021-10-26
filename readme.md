# Validation Free and Replication Robust Volume-based Data Valuation

This repository is the official implementation of the following paper accepted by the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS) 2021:

> Xinyi Xu*, Zhaoxuan Wu*, Chuan Sheng Foo, Bryan Kian Hsiang Low
>
> Validation Free and Replication Robust Volume-based Data Valuation

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
