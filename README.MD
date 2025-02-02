# CoDe: Blockwise Control for Denoising Diffusion Models

This is the official repository for the paper "CoDe: Blockwise Control for Denoising Diffusion Models"


## Overview
Aligning diffusion models to downstream tasks often requires finetuning new models or gradient-based guidance at inference time to enable sampling from the reward-tilted posterior. In this work, we explore a simple inference-time gradient-free guidance approach, called controlled denoising (CoDe), that circumvents the need for differentiable guidance functions and model finetuning. CoDe is a blockwise sampling method applied during intermediate denoising steps, allowing for alignment with downstream rewards. Our experiments demonstrate that, despite its simplicity, CoDe offers a favorable trade-off between reward alignment, prompt instruction following, and inference cost, achieving a competitive performance against the state-of-the-art baselines.

## Installation
This code is written in `Python 3.8.5` and requires the packages listed in [`environment.yml`](environment.yml).

To run the code, set up a virtual environment using `conda`:

```
cd <path-to-cloned-directory>
conda env create --file environment.yml
conda activate code
```
## Running experiments with CoDe and CoDe($\eta$)

To run CoDe or CoDe($\eta$), create a new configuration file in the [`configs`](BoN/configs/) directory. We provide the config files for CoDe and CoDe($\eta$). Additional details about each parameter can be found in [`template.yaml`](BoN/configs/template.yaml). 

The experiments can be run using the following command:

```
cd <path-to-cloned-directory>/BoN/src_sd
python  inference.py --config ../configs/<config-name>.yaml
```
We also provide the code to reproduce the results of the toy setup in [`gmm.ipynb`](BoN/gmm.ipynb)

## Other baselines and metrics

We also provide the code used to generate results of universal guidance and the code to compute performance metrics. The steps to run can be found within the respective directories.

- [`Universal Guidance`](Universal-Guided-Diffusion/README.md)
- [`T-CLIP`](clipscore/README.md)
- [`I-Gram`](CSD/README.md)
- [`FID`](pytorch-fid/README.md)
- [`CMMD`](cmmd-pytorch/README.md)
