"""
We donwload the pretrained embeddings from 
"""
import ipdb
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances


def load():
	dir_pretrained_model = Path("results/pretrained_tensorflow_model")
	fname_embeds = dir_pretrained_model / "test_vqvec2_flat.npy"
	fname_labels = dir_pretrained_model / "test_label_nucenter.csv"
	embeds = np.load(fname_embeds)
	df_labels = pd.read_csv(fname_labels)  # one per crop
	labels_prots = df_labels['name'].values.astype(str)
	labels_loc1 = df_labels['loc_grade1'].values.astype(str)

	# get consensus
	embeds_consensus, prots = get_consensus_encoding(embeds,
													 labels_prots,
													 sort_key=None)
	df_lookup = df_labels.groupby('name').sample(n=1)
	assert np.all(prots == df_lookup.name.values)

	return embeds_consensus, df_lookup


def get_consensus_encoding(embed_data, labels, sort_key=None):
	"""  """
	labels_uniq = np.array(sorted(np.unique(labels), key=sort_key))  # (1311,)

	embeds_consensus = []
	for label in labels_uniq:
		idxs = np.where(labels == label)[0]
		embed = embed_data[idxs].mean(0)
		embeds_consensus.append(embed)
	embeds_consensus = np.vstack(embeds_consensus)

	return embeds_consensus, labels_uniq

def get_dist_Science_paper_workflow(embeds_consensus,
					  df_lookup,
					  mode='correlation',
					  pca_dim=-1):
	from science_paper_repo.scripts.cytoself_analysis import clustering_workflows
	pass


def get_nearest_prots(embeds_consensus,
					  df_lookup,
					  mode='correlation',
					  pca_dim=-1,
					  target_protein='C4orf32',
					  get_rank=['STT3A', 'STT3B', 'OSTC', 'DAD1', 'KRTCAP2']):
	"""
	Args:
		pca_dim (int): if <1, then no PCA
	"""
	# target_protein='C4orf32'
	if pca_dim > 0:
		from sklearn.decomposition import PCA
		pca = PCA(n_components=200)
		X = pca.fit_transform(embeds_consensus)
		embeds_consensus = X
		raise NotImplementedError()

	labels_protein = df_lookup['name'].values
	labels_loc1 = df_lookup['loc_grade1'].values
	lookup_prot_to_locgrade1 = dict(
		zip(df_lookup['name'].values,
			df_lookup['loc_grade1'].values))
	dist = pairwise_distances(embeds_consensus,
							  embeds_consensus,
							  metric=mode)
	# query the nearest stuff
	idx = np.where(labels_protein == target_protein)[0]
	assert len(idx) == 1
	dist_argsort_target = np.argsort(dist[idx[0]])
	dists_nearest = dist[idx[0]][dist_argsort_target]
	assert dist_argsort_target[0] == idx[0]
	prots_nearest = labels_protein[dist_argsort_target]
	locgrade1_nearest = np.array(
		[lookup_prot_to_locgrade1[k] for k in prots_nearest])

	# print out the nearest 20 
	n_print=20
	print(prots_nearest[:n_print])
	print(locgrade1_nearest[:n_print])
	print(1-dists_nearest[:n_print])

	ipdb.set_trace()

	# compare a specific list of proteins that we expect to be close
	for prot in get_rank:
		idx = np.where(prot == prots_nearest)[0] 
		print(prot, idx, locgrade1_nearest[idx])

	if 0:
		is_er = np.array(['er' in s for s in labels_loc1.astype(str)])
		is_mito = np.array(['mito' in s for s in labels_loc1.astype(str)])
		is_vesicle = np.array(['vesicle' in s for s in labels_loc1.astype(str)])


if __name__ == "__main__":

	embeds_consensus, df_lookup = load()
	get_nearest_prots(embeds_consensus, df_lookup)
