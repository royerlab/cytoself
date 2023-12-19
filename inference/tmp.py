"""
Run in project home dir as 
	python inference/compare_opencell_targets.py

For a pretrained model in `<pretrained_model_dir>`, read in the embeddings for 
that model on the OpenCell dataset. This is the dataset the model was trained on

Next get the new inference dataset, its crops, and the embeddings that were 
computed in `inference/get_crop_features.py`. These embedding should have been 
taken from model from `<pretrained_model_dir>`. 
"""

import ipdb
import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
from typing import Literal


def get_nearest_proteins(fname_crops,
						 fname_crops_meta,
						 fname_annotations,
						 dir_pretrained_model,
						 mode='correlation'):
	# results dirs
	embed_opencell, labels_opencell, crops_opencell, df_opencell, df_opencell_lookup = get_embeds_and_crops(dir_pretrained_model,
						 dataset="opencell")

	# get inference embeddings for the new dataset
	embed_inf, labels_inf, crops_inf, df_meta_inf, df_annotations = get_embeds_and_crops(dir_pretrained_model, dataset='inference')

	## get "consensus encodings" - this one is the elementwise mean of the vqvec2 vecotors
	protein_label_opencell = labels_opencell[:, 1]
	# label for doing consensus in opencell is protein id
	embeds_consensus_opencell, labels_consensus_opencell = get_consensus_encoding(
		embed_opencell, protein_label_opencell)
	# label for doing consensus in inference is the well id. Multiple wells have the same protein
	protein_label_inf = df_meta_inf['well_id'].values
	embeds_consensus_inf, labels_consensus_inf = get_consensus_encoding(
		embed_inf, protein_label_inf, sort_key=custom_sort_key)
	# check that the labels for the consensus embeddings are sorted the same as their lookup dataframes
	assert np.all(
		df_opencell_lookup['prot_id'].values == labels_consensus_opencell)
	assert np.all(df_inf_lookup['well_id'].values == labels_consensus_inf)

	## do the whole analysis for mode 'correlation'
	df_results_summary = find_nearest_analysis(embeds_consensus_opencell,
											   labels_consensus_opencell,
											   embeds_consensus_inf,
											   labels_consensus_inf,
											   df_opencell_lookup,
											   df_inf_lookup,
											   dir_results,
											   dir_viz,
											   mode='correlation',
											   k_nearest_prots=10)
	visualize_knns(df_results_summary,
				   crops_inf,
				   df_meta_inf,
				   crops_opencell,
				   df_opencell,
				   dir_viz,
				   n_samples=10,
				   k_nearest_prots=10)

	## now do the whole analysis for mode 'hist'
	pass


def get_embeds_and_crops(dir_pretrained_model,
						 dataset: Literal['opencell',
										  'inference'] = "opencell"):
	"""
	Get cytoself embeddings and image crops for either opencell or the inference dataset. 
	Args: 
		'dataset' (str): one of 'opencell'
	"""
	print(f"loading dataset embeddings and crops [{dataset}]")

	if dataset == "opencell":
		# get the presaved embeddings from OpenCell
		f_embed = Path(dir_pretrained_model) / "embeddings/vqvec2.npy"
		embed_opencell = np.load(f_embed)  # (93037, 64, 4, 4)
		embed_opencell = torch.tensor(embed_opencell).view(len(embed_opencell), -1)
		dir_opencell_test = Path("data/test_dataset_metadata")
		crops_opencell = np.load(dir_opencell_test /
								 "test_dataset_crops.npy")  # (93037, 3, 100, 100)
		crops_opencell = crops_opencell[:, :
										2]  # if there is a nuclear distance channel, then remove it
		labels_opencell = np.load(
			dir_opencell_test /
			"test_dataset_labels.npy")  # (93037, 3) protid, prot, loc_grade1
		df_opencell = pd.DataFrame(labels_opencell,
								   columns=['ensg_id', 'prot_id', 'loc_grade1'])
		df_opencell_lookup = pd.DataFrame(
			np.unique(labels_opencell, axis=0),
			columns=['ensg_id', 'prot_id', 'loc_grade1']).sort_values('prot_id')

		return embed_opencell, labels_opencell, crops_opencell, df_opencell, df_opencell_lookup


	## get inference embeddings for the new dataset
	elif dataset == "inference":
		dir_embeddings = Path(
			"inference/results/get_crop_features") / dir_pretrained_model
		f_embeddings_vqvec2 = dir_embeddings / "embeddings_vqvec2.pt"
		embed_inf, labels_inf = torch.load(f_embeddings_vqvec2)
		embed_inf = torch.tensor(embed_inf).view(len(embed_inf), -1)
		df_meta_inf = pd.read_csv(dir_embeddings / "crops_meta.csv")
		df_annotations = pd.read_csv(fname_annotations)
		crops_inf = torch.load(fname_crops)
		crops_inf = T.Resize(100)(crops_inf)

		return embed_inf, labels_inf, crops_inf, df_meta_inf, df_annotations


	else:
		raise ValueError(f"dataset argument was [{dataset}]")



def find_nearest_analysis(embeds_consensus_opencell,
						  labels_consensus_opencell,
						  embeds_consensus_inf,
						  labels_consensus_inf,
						  df_opencell_lookup,
						  df_inf_lookup,
						  dir_results,
						  dir_viz,
						  mode='correlation',
						  k_nearest_prots=10):
	"""
	Take the consensus embeddings for opencell and inference data. For each inf 
	well_id, get the nearest `k_nearest_prots` proteins in Opencell. 
	The option `mode`, controls how 'nearest' is defined:
		`mode==correlation` is wrt vqvec2. 
	"""
	## compute the distance matrix between the inf proteins and the 1311 OpenCell proteins
	if mode == "correlation":
		dist = pairwise_distances(embeds_consensus_inf,
								  embeds_consensus_opencell,
								  metric='correlation')
	else:
		raise
	dist = torch.from_numpy(dist)
	# matrix (n_well_ids_inf, n_prots_opencell), which here is (80,1311)
	dist_argsort = torch.argsort(dist, axis=1)[:, :k_nearest_prots]

	prots_nearest = df_opencell_lookup['prot_id'].values[
		dist_argsort]  # (n_well_ids_inf,k_nearest_prots)
	loc_grade1_nearest = df_opencell_lookup['loc_grade1'].values[
		dist_argsort]  # (n_well_ids_inf,k_nearest_prots)

	## stick the knn preds together. Note that the rows of `df_inf_lookup` are aligned
	# with the rows of `prots_nearest` and `loc_grade1_nearest`. This was checked in
	# the function caller (it should check that `df_inf_lookup` algin with `labels_consensus_inf`
	df_results_summary = df_inf_lookup[[
		'well_id', 'protein', 'orgIP_cluster_annotation',
		'plate33_eyeballed_annotation'
	]]
	df_results_summary.index = range(len(df_results_summary))
	df_loc_grade1_nearest = pd.DataFrame(
		loc_grade1_nearest,
		columns=[
			f'loc_grade1_nn{i}' for i in range(loc_grade1_nearest.shape[1])
		])
	df_prot_nearest = pd.DataFrame(
		prots_nearest,
		columns=[f'prot_nn{i}' for i in range(prots_nearest.shape[1])])
	df_results_summary = pd.concat(
		[df_results_summary, df_loc_grade1_nearest, df_prot_nearest], axis=1)

	# save the 'knn predictions'
	f_save = dir_results / f"knn_opencell_preds_mode_{mode}.csv"
	df_results_summary.to_csv(f_save)

	return df_results_summary


def norm_0_1_perimg_perch(crops):
	""" 
	input is (n,nchanells,Y,X)
	Normalize to [0,1] within each image and within each channel. 
	"""
	if type(crops) == np.ndarray:
		crops = torch.from_numpy(crops)
	crops_view = crops.view((*crops.shape[:2], -1))
	l = crops_view.min(dim=-1)[0]
	l = l.unsqueeze(-1).unsqueeze(-1)
	u = crops_view.max(dim=-1)[0]
	u = u.unsqueeze(-1).unsqueeze(-1)
	crops = (crops - l) / (u - l)

	return crops


def visualize_knns(df_results_summary,
				   crops_inf,
				   df_inf,
				   crops_opencell,
				   df_opencell,
				   dir_viz,
				   n_samples=10,
				   k_nearest_prots=10):
	""" For each inference protein, create a lookup.  """

	np.random.seed(0)
	for idx, row in df_results_summary.iterrows():
		imgs = []
		well_id = row['well_id']
		protein = row['protein']
		idxs_img = np.where(df_inf['well_id'] == well_id)[0]
		np.random.shuffle(idxs_img)
		imgs.append(crops_inf[idxs_img[:n_samples]])

		# get the nearest protein set
		prots_nearest = []
		for j in range(k_nearest_prots):
			prot = row[f'prot_nn{j}']
			prots_nearest.append(prot)
			idxs_img = np.where(df_opencell['prot_id'] == prot)[0]
			np.random.shuffle(idxs_img)
			imgs.append(torch.from_numpy(crops_opencell[idxs_img[:n_samples]]))

		# if 1:
		# 	imgs_tmp = crops_opencell[:5]
		# 	imgs_tmp = norm_0_1_perimg_perch(torch.tensor(imgs_tmp))
		# 	f,axs = plt.subplots(1,3)
		# 	[axs[i].imshow(imgs_tmp[0,i]) for i in range(3)]
		# 	f.savefig(dir_viz / "tmp.png", dpi=200)

		imgs = torch.cat(imgs)
		imgs = norm_0_1_perimg_perch(imgs)
		imgs_show = torch.zeros(len(imgs),
								3,
								*imgs.shape[-2:],
								dtype=imgs.dtype)
		imgs_show[:, 1] = imgs[:, 0]  # protein in green channel
		imgs_show[:, 2] = imgs[:, 1]  # nucleus in green channel
		grid = make_grid(imgs_show, nrow=n_samples).permute(1, 2, 0)
		f_save = dir_viz / f"knn_samples_wellid_{well_id}_prot_{protein}"
		title = f"well_{well_id}_prot_{protein}\nnearest={prots_nearest}"
		f, axs = plt.subplots(figsize=(15, 15))
		axs.imshow(grid)
		axs.set(title=title)
		f.savefig(f_save, dpi=200)
		plt.close()

	# ipdb.set_trace()

	pass


def get_consensus_encoding(embed_data, labels, sort_key=None):
	"""  """
	labels_uniq = np.array(sorted(np.unique(labels), key=sort_key))  # (1311,)

	embeds_consensus = []
	for label in labels_uniq:
		idxs = np.where(labels == label)[0]
		embed = embed_data[idxs].mean(0)
		embeds_consensus.append(embed)
	embeds_consensus = torch.vstack(embeds_consensus)

	return embeds_consensus, labels_uniq


def custom_sort_key(x):
	"""
	in `df_inf_lookup`, sort the wellids ('A1','A2','A10'...) according to 
	the letter and then the number, where you treat the number as an int and not a string
	So, split the well_id into a prefix (e.g., 'A') and a numeric part 
	(e.g., '1', '3', '10', '11')
	"""
	prefix, numeric = x[0], int(x[1:])
	return (prefix, numeric)


def get_df_inf_lookup(df_meta_inf):
	"""
	`df_meta_inf` is a row for each crop. Each crop from the same well_id has 
	identical rows. This function returns a lookup function that is well_id 
	"""
	df_inf_lookup = df_meta_inf.groupby('well_id').sample(
		n=1).sort_values('well_id')
	df_inf_lookup['sort_key'] = df_inf_lookup['well_id'].apply(custom_sort_key)
	df_inf_lookup = df_inf_lookup.sort_values(by='sort_key').drop(
		columns=['sort_key'])
	df_inf_lookup.index = df_inf_lookup['well_id']

	return df_inf_lookup


if __name__ == "__main__":
	fname_crops = "inference/results/crop/crops.pt"
	fname_crops_meta = "inference/results/crop/crops_meta.csv"
	# this file is protein ids and estimates of their annotations
	fname_annotations = "data/cz_infectedcell_finalwellmapping.csv"
	dir_pretrained_model = "results/20231011_train_all_no_nucdist"

	# let's look at statistics for cytoself stuff

	# the main prediction function
	get_nearest_proteins(fname_crops=fname_crops,
						 fname_crops_meta=fname_crops_meta,
						 fname_annotations=fname_annotations,
						 dir_pretrained_model=dir_pretrained_model)
