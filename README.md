# cytoself

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![PyPI](https://img.shields.io/pypi/v/cytoself.svg)](https://pypi.org/project/cytoself)
[![Python Version](https://img.shields.io/pypi/pyversions/cytoself.svg)](https://python.org)
[![DOI](http://img.shields.io/badge/DOI-10.1101/2021.03.29.437595-B31B1B.svg)](https://doi.org/10.1101/2021.03.29.437595)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)


![Alt Text](images/rotating_umap.gif)

cytoself is a self-supervised platform that we developed for learning features of protein subcellular localization from cell images. 
This model is described in detail in our recent preprint [[2]](https://www.biorxiv.org/content/10.1101/2021.03.29.437595v1).
The representations derived from cytoself encapsulate highly specific features that can derive functional insights for 
proteins on the sole basis of their localization.

Applying cytoself to images of endogenously labeled proteins from the recently released 
[OpenCell](https://opencell.czbiohub.org) database creates a highly resolved protein localization atlas
[[1]](https://www.biorxiv.org/content/10.1101/2021.03.29.437450v1). 

[1] Cho, Nathan H., et al. "OpenCell: proteome-scale endogenous tagging enables the cartography of human cellular organization." bioRxiv (2021).
https://www.biorxiv.org/content/10.1101/2021.03.29.437595v1 <br />
[2] Kobayashi, Hirofumi, et al. "Self-Supervised Deep-Learning Encodes High-Resolution Features of Protein Subcellular Localization." bioRxiv (2021).
https://www.biorxiv.org/content/10.1101/2021.03.29.437595v1


## How cytoself works
cytoself uses images and its identity information as a label to learn the localization patterns of proteins.
We used cell images where single protein is labeled and the ID of labeled protein as identity information.

![Alt Text](images/workflow.jpg)


## What's in this repository
This repository offers three main components: `DataManager`, `cytoself.models`, and `Analytics`.

`DataManager` is a simple module to handle train, validate and test data. 
You may want to modify it to adapt to your own data structure.
This module is in `cytoself.data_loader.data_manager`.

`cytoself.models` contains modules for three different variants of the cytoself model: 
a model without split-quantization, a model without the pretext task, and the 'full' model (refer to our preprint for details about these variants). 
There is a submodule for each model variant that provides methods for constructing, compiling, and training the models (which are built using tensorflow).

`Analytics` is a simple module to perform analytic processes such as dimension reduction and plotting. 
You may want to modify it too to perform your own analysis. This module is in `cytoself.analysis.analytics`. 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/royerlab/cytoself_internal/blob/main/examples/simple_example.ipynb)


## Installation
Recommended: create a new environment by
```shell script
conda create -n cytoself python=3.7
```

Clone this repository 

```shell script
git clone https://github.com/royerlab/cytoself.git
```

## (Option) Install TensorFlow GPU
If your computer is equipped with GPUs that supports Tensorflow 1.15, you can install Tensorflow-gpu to utilize GPUs.
Make sure to install the following packages before running `setup.py`, otherwise you may want to uninstall and reinstall them with conda.
```shell script
conda install h5py=2.10.0
conda install tensorflow-gpu=1.15
```

Run the following code inside the cytoself folder to install dependencies.

```shell script
python setup.py develop
```

## Example script
A minimal example script is in `example/simple_training.py`.

Test if this package runs in your computer with command 
```shell script
python examples/simple_example.py
```


## Computation resources
It is highly recommended to use GPU to run cytoself. 
A full model with image shape (100, 100, 2) and batch size 64 can take ~9GB of GPU memory.


## Tested Environment
Google Colab (CPU/GPU/TPU)

macOS 10.14.6, RAM 32GB (CPU)

Windows10 Pro 64bit, RAM 32GB (CPU)

Ubuntu 18.04.5 LTS, TITAN xp, CUDA 10.2 (GPU)


