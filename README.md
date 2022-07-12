# cytoself_pytorch
cytoself in pytorch implementation

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-397/)
[![DOI](http://img.shields.io/badge/DOI-10.1101/2021.03.29.437595-B31B1B.svg)](https://doi.org/10.1101/2021.03.29.437595)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![codecov](https://codecov.io/gh/royerlab/cytoself_pytorch/branch/main/graph/badge.svg?token=2SMIDRRC5L)](https://codecov.io/gh/royerlab/cytoself_pytorch)
[![Tests](https://github.com/royerlab/cytoself_pytorch/actions/workflows/pytest-codecov-conda.yml/badge.svg)](https://github.com/royerlab/cytoself_pytorch/actions/workflows/pytest-codecov-conda.yml)


![Alt Text](images/rotating_umap.gif)

## Installation
Recommended: create a new environment and install cytoself on the environment from pypi
```shell script
conda create -y -n cytoself python=3.9
conda activate cytoself
pip install cytoself
```

### (For the developers)
### How to install dependencies

```bash
pip install -r requirements/requirements.txt
```

### How to install development dependencies

```bash
pip install -r requirements/development.txt
```

## How to use on the example data 
Download one set of the image and label data from [Data Availability](##Data Availability)


### 1. Prepare Data
```python
from cytoself.analysis.analysis_opencell import AnalysisOpenCell
from cytoself.datamanager.datamanager_oc import DataManagerOpenCell
from cytoself.trainer.cytoselflight_trainer import CytoselfLiteTrainer

datapath = 'path/to/the/data'
datamanager = DataManagerOpenCell(datapath, ['gfp'], batch_size=32)
datamanager.const_dataset(num_labels=3)
datamanager.const_dataloader()
```

### 2. Create and train a cytoself model
```python
model_args = {
    'input_shape': (1, 100, 100),
    'emb_shapes': ((64, 25, 25), (64, 4, 4)),
    'output_shape': (1, 100, 100),
    'vq_args': {'num_embeddings': 512},
    'num_class': 3,
}
train_args = {
    'lr': 1e-3,
    'max_epochs': 90,
    'reducelr_patience': 3,
    'reducelr_increment': 0.1,
    'earlystop_patience': 6,
}
trainer = CytoselfLiteTrainer(model_args, train_args, homepath='demo_output')
trainer.fit(datamanager, tensorboard_path='tb_logs')
```

### 3. Plot UMAP
```python
from os.path import join

analysis = AnalysisOpenCell(datamanager, trainer)
umap_data = analysis.plot_umap_of_embedding_vector(
    data_loader=datamanager.test_loader,
    savepath='path/to/save/fig/umap_vqvec2.png',
    group_col=1,
    output_layer=f'vqvec2',
    title=f'UMAP vqvec2',
    xlabel='x axis',
    ylabel='y axis',
    s=0.3,
    alpha=0.5,
    show_legend=True,
)
```


## Tested Environment
~~Google Colab (CPU/GPU/TPU)~~

~~macOS 10.14.6, RAM 32GB (CPU)~~

~~Windows10 Pro 64bit, RAM 32GB (CPU)~~

Ubuntu 20.04.3 LTS, NVIDIA 3090, CUDA 11.4 (GPU)

## Data Availability
The full data used in this work can be found here.
The image data have the shape of `[batch, 100, 100, 4]`, in which the last channel dimension corresponds to `[target 
protein, nucleus, nuclear distance, nuclear segmentation]`.

Due to the large size, the whole data is split to 10 files. The files are intended to be concatenated together to 
form one large numpy file or one large csv.

[Image_data00.npy](https://drive.google.com/file/d/15_CHBPT-p5JG44acP6D2hKd8jAacZatp/view?usp=sharing)  
[Image_data01.npy](https://drive.google.com/file/d/1m7Cj2OALiZTIiHpvb9zFPG_I3j1wRnzK/view?usp=sharing)  
[Image_data02.npy](https://drive.google.com/file/d/17nknzqlcYO3n9bAe4FwGVPkU-mJAhQ4j/view?usp=sharing)  
[Image_data03.npy](https://drive.google.com/file/d/1vEsddF68dyOda-hwI-ptAL4vShBGl98Y/view?usp=sharing)  
[Image_data04.npy](https://drive.google.com/file/d/1aB7WaRuhobG_IDl0l_PPeSJAxCYy-Pye/view?usp=sharing)  
[Image_data05.npy](https://drive.google.com/file/d/1qb0waKcLprDtuFAdCec3WegWkmd-U45A/view?usp=sharing)  
[Image_data06.npy](https://drive.google.com/file/d/1y-1vlfZ4eNhvTvpuqTZVL8DvSwYX3CH_/view?usp=sharing)  
[Image_data07.npy](https://drive.google.com/file/d/1ejcPdh-d5lB1OcZ6x8SJx61pEUioZvB2/view?usp=sharing)  
[Image_data08.npy](https://drive.google.com/file/d/1DOicAkruNsU5F4DWLzO2QrV6xU4kuVxs/view?usp=sharing)  
[Image_data09.npy](https://drive.google.com/file/d/1a5YyHeRSRdJStG3KnFe2vsNjrsit9zbf/view?usp=sharing)  
[Label_data00.csv](https://drive.google.com/file/d/1CVwvXW2KhVBbTBixwRXIIiMhrlGDXz-4/view?usp=sharing)  
[Label_data01.csv](https://drive.google.com/file/d/1mTYe5icvWXNfY5wEsuQUhSwgtefBJpjg/view?usp=sharing)  
[Label_data02.csv](https://drive.google.com/file/d/1HckmktklyPo6qbakrwtERsCT34mRdn7l/view?usp=sharing)  
[Label_data03.csv](https://drive.google.com/file/d/1GBxDmWcl_o49i4lGujA8EgIn5G4htkBr/view?usp=sharing)  
[Label_data04.csv](https://drive.google.com/file/d/1G4FpJnlqB3ejmdw3SF2w3DFYt8Wnq0fT/view?usp=sharing)  
[Label_data05.csv](https://drive.google.com/file/d/1Vo1J09qP2TAoXwltCF84socz2TPV92JU/view?usp=sharing)  
[Label_data06.csv](https://drive.google.com/file/d/1d7gJjLTQhOw-e9KZJY9pr6KOCIN8NBvp/view?usp=sharing)  
[Label_data07.csv](https://drive.google.com/file/d/1kr5EF0RA3ZwSXmoaBFwFDVnrokh2EaOE/view?usp=sharing)  
[Label_data08.csv](https://drive.google.com/file/d/1mXyedmLezzty2LSSH3asw0LQeu-ie9mz/view?usp=sharing)  
[Label_data09.csv](https://drive.google.com/file/d/1Vdv1cD75VhvC3FdKTen-5rqLJnWpHvmb/view?usp=sharing)  
