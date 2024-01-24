"""
Based on the Science analysis in 
https://nbviewer.org/github/czbiohub/2021-opencell-figures/blob/master/notebooks/localization_clustering/clustering-performance.ipynb#Plots-of-ARI-vs-Leiden-resolution 

Compute the nearest neighbour graph for Leiden clustering with the Euclidean 
metric. 
"""
import ipdb
import anndata as ad
import datetime
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path
import scanpy as sc
import seaborn as sns
import sys
import os
import torch

import sklearn.manifold
import sklearn.metrics
import sklearn.neighbors
from sklearn.metrics import pairwise_distances

from matplotlib import pyplot as plt
from matplotlib import rcParams

current_filename = Path(os.path.basename(__file__))

# this needs to git pull from https://github.com/czbiohub-sf/2021-opencell-figures/tree/master
# then use path append to add to i t
sys.path.insert(0, '../../2021-opencell-figures')
import scripts.cytoself_analysis.clustering_workflows
import scripts.cytoself_analysis.ground_truth_labels
import scripts.cytoself_analysis.go_utils
from scripts.cytoself_analysis import (clustering_workflows,
                                       ground_truth_labels, go_utils)

# hyperparams from the OG notebook
n_neighbors = 10
n_pcs = 200
metric = 'euclidean'

sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
rcParams['font.family'] = 'sans-serif'
rcParams['axes.grid'] = False

data_dir = Path('inference/results/compare_opencell_targets/')
output_dir = pathlib.Path('inference/results/clustering_leiden/')
output_dir.mkdir(exist_ok=True, parents=True)


def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d')


# opencell labels
dir_opencell_test = Path("data/test_dataset_metadata")
labels_opencell_crops = np.load(
    dir_opencell_test /
    "test_dataset_labels.npy")  # (93037, 3) protid, prot, loc_grade1
df_opencell_lookup = pd.DataFrame(np.unique(labels_opencell_crops, axis=0),
                                  columns=['ensg_id', 'prot_id', 'loc_grade1'
                                           ]).sort_values('prot_id')

# generate_inference_crop_grids("results/20231218_train_all_no_nucdist_balanced_classes_2",
# fname_crops, nsamples=100, ncols=10)
dir_pretrained_models_checkpoints = [
    # ["results/20231218_train_all_balanced_classes_2", None],
    ["results/20231218_train_all_no_nucdist_balanced_classes_2", None],
    # ["results/20231222_train_all_balanced_classes_1", None],
    # ["results/20231222_train_all_no_nucdist_balanced_classes_1", None],
    # ["results/20231222_train_all_balanced_classes_1", None],
    # ["results/20231221_train_with_orphans", None],
    # ["results/20231221_train_with_orphans_no_nucdist", None],
    # ["results/20231222_train_all_no_nucdist_balanced_classes_1", None],
    # ["results/20231222_train_all_balanced_classes_1", None],
    # ["results/20231218_train_all_no_nucdist", None],
    # ["results/20231022_train_all", None],
]
for (dir_pretrained_model, checkpoint) in dir_pretrained_models_checkpoints:

    dir_results = Path("inference/results/compare_opencell_targets/"
                       ) / dir_pretrained_model / f"ckpt_{checkpoint}"
    data_opencell = torch.load(dir_results / "consensus_embeddings.pt").numpy()
    prots_opencell = torch.load(dir_results / "consensus_labels.pt")
    assert all(labels_opencell == df_opencell_lookup['prot_id'].values)

    data_inf = torch.load(dir_results /
                          "consensus_embeddings_inference.pt").numpy()
    prots_inf = torch.load(dir_results / "consensus_labels_inference.pt")

    data = np.concatenate((data_opencell, data_inf))
    prots_all = np.concatenate((prots_opencell, prots_inf))
    labels_loc_grade1 = np.concatenate((df_opencell_lookup["loc_grade1"],
                                        np.array(["ORF"] * len(prots_inf))))

    adata = ad.AnnData(data)
    cwv = clustering_workflows.ClusteringWorkflow(adata=adata)
    cwv.calculate_neighbors(n_neighbors=n_neighbors,
                            n_pcs=n_pcs,
                            metric=metric)

    resolution = 1 # 0.63  # 0.63
    random_state = 0
    cwv.run_leiden(resolution=resolution, random_state=random_state)
    y_leiden = np.array(adata.obs.leiden)
    
    # print out some samples from each cluster to check it's coherent 
    n_samples = 50
    y='7'
    for y in np.unique(y_leiden):
        idxs = np.where(y_leiden == y)[0]
        print(f"Label: {y}, num_prots: {len(idxs)}")
        print(f"Annots: ")
        print(labels_loc_grade1[idxs[:n_samples]])
        print(prots_all[idxs[:n_samples]])
        print()

    ipdb.set_trace()

prot = "TMEM184C" # "ANKRD46"
idx = np.where(labels == prot)[0] # idx = 1311
neighbors = adata.obsp['distances'][idx].indices
prots = labels[[neighbors]]

dist = pairwise_distances(data, data, metric=metric)
dist = torch.from_numpy(dist)
dist_argsort = torch.argsort(dist, axis=1)
