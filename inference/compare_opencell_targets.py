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
from tqdm import tqdm
from collections import Counter
from sklearn.decomposition import PCA

protein_filter = dict(
    # GLT8D1=[1, 2, 4, 11, 13, 18, 52, 64, 96, 97],
    GLT8D1=[2, 4, 11, 18, 52, 64, 96, 97],
    # GLT8D1=[97],
    MFSD5=[
        0, 2, 8, 12, 13, 15, 21, 22, 24, 25, 40, 42, 44, 46, 48, 51, 57, 67,
        68, 71, 73, 75, 77, 78, 86, 89, 92, 94, 96, 97, 98, 99
    ],
    MPZL1=[
        19, 24, 26, 27, 39, 35, 36, 39, 40, 41, 42, 45, 46, 47, 48, 49, 50, 51,
        53, 54, 55, 56, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 85, 86, 87,
        88, 89, 91, 93, 94, 95, 97
    ],
    TSPAN3=[
        0, 1, 23, 8, 9, 10, 11, 12, 13, 15, 19, 21, 22, 29, 30, 31, 32, 33, 34,
        37, 39, 40, 42, 43, 44, 45, 49, 51, 52, 54, 63, 64, 90, 92, 98
    ],
)

current_filename = Path(os.path.basename(__file__))
mito_proteins = np.array([
    'ACAP3', 'DNAJC11', 'PKBP8', 'MFN1', 'MFN2', 'MYO19', 'RAB24', 'SAR1A',
    'TOMM20', 'TOMM70', 'VDAC1', 'VDAC2', 'VDAC3'
])


def get_nearest_proteins(
        fname_crops,
        fname_crops_meta,
        fname_annotations,
        dir_pretrained_model,
        pca_dim=200,
        mode='correlation',
        representation='vqvec2',  # or 'hist'
        combine_inference_wells=True,
        do_filtering=False,
        do_visualize_knns=True,
        checkpoint=None,
        do_include_nn_losses=False,
        combine_augmented_embeddings=False,
        test=None):
    """ 
	deorphaning analysis 
	Args: 
		do_filtering (bool).
	"""
    dir_results = Path(
        "inference/results"
    ) / current_filename.stem / dir_pretrained_model / f"ckpt_{checkpoint}"
    dir_results.mkdir(exist_ok=True, parents=True)

    dir_viz = dir_results

    # results dirs
    embed_opencell, labels_opencell, crops_opencell, df_opencell, df_opencell_lookup = get_embeds_and_crops(
        dir_pretrained_model,
        checkpoint=checkpoint,
        representation=representation,
        combine_augmented_embeddings=combine_augmented_embeddings,
        dataset="opencell")
    lookup_prot_to_locgrade1 = dict(
        zip(df_opencell_lookup['prot_id'].values,
            df_opencell_lookup['loc_grade1'].values))

    # get inference embeddings for the new dataset
    embed_inf, labels_inf, crops_inf, df_meta_inf, df_annotations, df_inf_lookup = get_embeds_and_crops(
        dir_pretrained_model,
        checkpoint=checkpoint,
        representation=representation,
        combine_augmented_embeddings=combine_augmented_embeddings,
        dataset='inference')
    n_inf = len(df_inf_lookup)

    ## get "consensus encodings" - this one is the elementwise mean of the vqvec2 vecotors
    protein_label_opencell = labels_opencell[:, 1]
    # label for doing consensus in opencell is protein id
    embeds_consensus_opencell, labels_consensus_opencell = get_consensus_encoding(
        embed_opencell, protein_label_opencell)
    torch.save(embeds_consensus_opencell,
               dir_results / "consensus_embeddings.pt")

    lookup_loss_ce, lookup_loss_recon = get_opencell_protein_losses(
        dir_pretrained_model, checkpoint)

    # label for doing consensus in inference is the well id. Multiple wells have the same protein
    if combine_inference_wells:
        protein_label_inf = df_meta_inf['protein'].values
        sort_key = None
        df_inf_lookup = df_inf_lookup.sort_values(
            by='protein'
        ).iloc[::2]  # we'll have an assert check that this is fine
    else:
        protein_label_inf = df_meta_inf['well_id'].values
        sort_key = custom_sort_key

    # do the conensus embeddings
    embeds_consensus_inf, labels_consensus_inf = get_consensus_encoding(
        embed_inf,
        protein_label_inf,
        sort_key=sort_key,
        do_filtering=do_filtering,
        crops=crops_inf)

    # check that the labels for the consensus embeddings are sorted the same as their lookup dataframes
    # for inference dataset, if combine_inference_wells, sort by protein, otherwise sort by well id.
    assert np.all(
        df_opencell_lookup['prot_id'].values == labels_consensus_opencell)
    if combine_inference_wells:
        assert np.all(df_inf_lookup['protein'].values == labels_consensus_inf)
    else:
        assert np.all(df_inf_lookup['well_id'].values == labels_consensus_inf)

    if pca_dim is not None and pca_dim > 0:
        pca = PCA(n_components=pca_dim, random_state=0)
        pca.fit(embeds_consensus_opencell)
        embeds_consensus_opencell = pca.transform(embeds_consensus_opencell)
        embeds_consensus_inf = pca.transform(embeds_consensus_inf)

    # get distances
    dist = pairwise_distances(embeds_consensus_inf,
                              embeds_consensus_opencell,
                              metric=mode)
    dist = torch.from_numpy(dist)

    # matrix (n_well_ids_inf, n_prots_opencell), which here is (80,1311)
    dist_argsort = torch.argsort(dist, axis=1)

    # branch for manual checking
    if 1:
        target_protein = 'GLT8D1'
        idxs = np.where(df_inf_lookup['protein'] == target_protein)
        assert len(idxs) > 0
        idx = idxs[0]

        dist_target = dist[idx[0]]
        dist_argsort_target = np.argsort(dist_target)
        prots_nearest = labels_consensus_opencell[dist_argsort_target]
        dist_nearest = dist_target[dist_argsort_target]
        locgrade1_nearest = np.array(
            [lookup_prot_to_locgrade1[k] for k in prots_nearest])

        n_prots = 15
        for i in range(n_prots):
            print(
                f"{i:3} {prots_nearest[i]:8} {locgrade1_nearest[i]:22} {1-dist_nearest[i]:.3f} "
            )

    # this function is where the csv file gets saved
    fname_stem = f"filter_{do_filtering}__representation_{representation}__rotations_{combine_augmented_embeddings}__mode_{mode}__pcadim_{pca_dim}"
    df_results_summary = find_nearest_analysis(
        embeds_consensus_opencell,
        labels_consensus_opencell,
        embeds_consensus_inf,
        labels_consensus_inf,
        df_opencell_lookup,
        df_inf_lookup,
        dir_results,
        dir_viz,
        fname_stem,
        do_include_nn_losses=do_include_nn_losses,
        mode=mode,
        dist=dist,
        lookup_loss_ce=lookup_loss_ce,
        lookup_loss_recon=lookup_loss_recon,
        k_nearest_prots=8)
    if do_visualize_knns:
        visualize_knns(
            df_results_summary,
            crops_inf,
            df_meta_inf,
            crops_opencell,
            df_opencell,
            dir_viz,
            fname_stem,
            n_samples=10,
            k_nearest_prots=4,
        )


def nearest_prots_within_opencell(
        dir_pretrained_model,
        mode='correlation',
        target_protein='C4orf32',
        pca_dim=200,
        representation='vqvec2',
        get_rank=['STT3A', 'STT3B', 'OSTC', 'DAD1', 'KRTCAP2']):
    """
	Given a target_protein that is in OpenCell, get the nearest proteins that are also 
	in OPenCell (this does not support a separate orphan protein). 
	Args:
		target_protein (str): target protein whose neighbors we're looking for
		get_rank (List[str]): list of proteins whose position in the kNN list we want to query
	"""
    ## results dirs
    dir_results = Path(
        "inference/results") / current_filename.stem / dir_pretrained_model
    dir_results.mkdir(exist_ok=True, parents=True)
    dir_viz = Path("inference/viz") / current_filename.stem
    dir_viz.mkdir(exist_ok=True, parents=True)

    embed_opencell, labels_opencell, crops_opencell, df_opencell, df_opencell_lookup = get_embeds_and_crops(
        dir_pretrained_model,
        representation=representation,
        combine_augmented_embeddings=combine_augmented_embeddings,
        dataset="opencell")
    lookup_prot_to_locgrade1 = dict(
        zip(df_opencell_lookup['prot_id'].values,
            df_opencell_lookup['loc_grade1'].values))

    ## get "consensus encodings" - this one is the elementwise mean of the vqvec2 vecotors
    protein_label_opencell = labels_opencell[:, 1]
    # label for doing consensus in opencell is protein id
    embeds_consensus_opencell, labels_consensus_opencell = get_consensus_encoding(
        embed_opencell, protein_label_opencell)
    # distance matrix
    dist = pairwise_distances(embeds_consensus_opencell,
                              embeds_consensus_opencell,
                              metric='correlation')
    #
    target_protein = 'LAMP1'
    idx = np.where(labels_consensus_opencell == target_protein)[0]
    assert len(idx) == 1
    dist_target = dist[idx[0]]
    dist_argsort_target = np.argsort(dist_target)
    prots_nearest = labels_consensus_opencell[dist_argsort_target]
    dist_nearest = dist_target[dist_argsort_target]
    locgrade1_nearest = np.array(
        [lookup_prot_to_locgrade1[k] for k in prots_nearest])
    n_prots = 30
    for i in range(n_prots):
        print(
            f"{i:3} {prots_nearest[i]:8} {locgrade1_nearest[i]:22} {1-dist_nearest[i]:.3f} "
        )

    for prot in get_rank:
        idx = np.where(prot == prots_nearest)[0] - 1
        print(prot, idx, locgrade1_nearest[idx])


def get_embeds_and_crops(
        dir_pretrained_model,
        checkpoint,
        representation="vqvec2",  #("vqvec2",'hist')
        do_filtering=False,
        combine_augmented_embeddings=False,
        dataset: Literal['opencell', 'inference'] = "opencell"):
    """
	Get cytoself embeddings and image crops for either opencell or the inference dataset. 
	Args: 
		'dataset' (str): one of 'opencell'
		combine_inference_wells (bool): if True AND if dataset=='opencell', then 
			combine the encodings 
		do_filtering: for inferencendataset, filter the crops to only include the ones listed
	"""
    print(f"loading dataset embeddings and crops [{dataset}]")

    augmentations = ("0", "90", "180", "270", "0_f", "90_f", "180_f", "270_f")
    if not combine_augmented_embeddings:
        augmentations = augmentations[:1]

    if dataset == "opencell":
        # get the presaved embeddings from OpenCell
        # f_embed = Path(dir_pretrained_model) / "embeddings/vqvec2.npy"
        dir_embeddings = Path("inference/results/get_crop_features"
                              ) / dir_pretrained_model / f"ckpt_{checkpoint}"

        embeds = []
        for augmentation in augmentations:
            if representation == 'vqvec2':
                f_embed = dir_embeddings / f"embeddings_vqvec2_opencell_aug{augmentation}.pt"
            elif representation == 'hist':
                f_embed = dir_embeddings / f"embeddings_vqindhist1_opencell_aug{augmentation}.pt"
            else:
                raise
            embed_opencell, _ = torch.load(f_embed)
            embed_opencell = torch.tensor(embed_opencell).view(
                len(embed_opencell), -1)
            embeds.append(embed_opencell)
        embeds = torch.stack(embeds)

        dir_opencell_test = Path("data/test_dataset_metadata")
        crops_opencell = np.load(
            dir_opencell_test /
            "test_dataset_crops.npy")  # (93037, 3, 100, 100)
        crops_opencell = crops_opencell[:, :
                                        2]  # if there is a nuclear distance channel, then remove it
        labels_opencell = np.load(
            dir_opencell_test /
            "test_dataset_labels.npy")  # (93037, 3) protid, prot, loc_grade1
        df_opencell = pd.DataFrame(
            labels_opencell, columns=['ensg_id', 'prot_id', 'loc_grade1'])
        df_opencell_lookup = pd.DataFrame(
            np.unique(labels_opencell, axis=0),
            columns=['ensg_id', 'prot_id',
                     'loc_grade1']).sort_values('prot_id')

        return embeds, labels_opencell, crops_opencell, df_opencell, df_opencell_lookup

    ## get inference embeddings for the new dataset
    elif dataset == "inference":
        dir_embeddings = Path("inference/results/get_crop_features"
                              ) / dir_pretrained_model / f"ckpt_{checkpoint}"

        embeds = []
        for augmentation in augmentations:
            if representation == 'vqvec2':
                f_embed = dir_embeddings / f"embeddings_vqvec2_aug{augmentation}.pt"
            elif representation == 'hist':
                f_embed = dir_embeddings / f"embeddings_vqindhist1_aug{augmentation}.pt"
            else:
                raise

            embed_inf, labels_inf = torch.load(f_embed)
            embed_inf = torch.tensor(embed_inf).view(len(embed_inf), -1)

            embeds.append(embed_inf)
        embeds = torch.stack(embeds)

        df_meta_inf = pd.read_csv(dir_embeddings / "crops_meta.csv")
        df_annotations = pd.read_csv(fname_annotations)
        crops_inf = torch.load(fname_crops)
        crops_inf = T.Resize(100)(crops_inf)

        ## join the inference metadata with the csv that has their protids and annotations
        df_meta_inf['well_id'] = [
            Path(f).stem.split("_")[-3] for f in df_meta_inf['fname_pro']
        ]
        df_meta_inf['fov_id'] = [
            Path(f).stem.split("_")[-2] for f in df_meta_inf['fname_pro']
        ]
        df_meta_inf = df_meta_inf.merge(df_annotations,
                                        how='left',
                                        left_on='well_id',
                                        right_on='well_id_new')
        df_inf_lookup = get_df_inf_lookup(df_meta_inf)

        return embeds, labels_inf, crops_inf, df_meta_inf, df_annotations, df_inf_lookup

    else:
        raise ValueError(f"dataset argument was [{dataset}]")


def order_by_frequency_with_counts(strings):
    # Count the frequency of each string
    freq = Counter(strings)

    # Sort the items by frequency (descending) and then alphabetically
    sorted_items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

    return sorted_items


def get_opencell_protein_losses(dir_pretrained_model, checkpoint):
    """ 
	For the proteins in the OpenCell test set, measure the get the average 
	of the losses (the protein classification loss & the recon loss).
	"""
    fname_loss_ce = Path(
        "inference/results/get_crop_features"
    ) / dir_pretrained_model / f"ckpt_{checkpoint}" / "loss_ce.csv"
    loss_ce_per_protein = pd.read_csv(fname_loss_ce)
    lookup_loss_ce = dict(
        zip(loss_ce_per_protein['protein'].values,
            loss_ce_per_protein['loss_ce'].values))

    fname_loss_recon = Path(
        "inference/results/get_crop_features"
    ) / dir_pretrained_model / f"ckpt_{checkpoint}" / "loss_recon.csv"
    loss_recon_per_protein = pd.read_csv(fname_loss_recon)
    lookup_loss_recon = dict(
        zip(loss_recon_per_protein['protein'].values,
            loss_recon_per_protein['loss_recon'].values))

    return lookup_loss_ce, lookup_loss_recon


def find_nearest_analysis(embeds_consensus_opencell,
                          labels_consensus_opencell,
                          embeds_consensus_inf,
                          labels_consensus_inf,
                          df_opencell_lookup,
                          df_inf_lookup,
                          dir_results,
                          dir_viz,
                          fname_stem,
                          mode,
                          lookup_loss_ce,
                          lookup_loss_recon,
                          do_include_nn_losses=False,
                          dist=None,
                          k_nearest_prots=10,
                          return_dists_only=False):
    """
	Take the consensus embeddings for opencell and inference data. For each inf 
	well_id, get the nearest `k_nearest_prots` proteins in Opencell. 
	The option `mode`, controls how 'nearest' is defined:
		`mode==correlation` is wrt vqvec2. 

	If `return_dists_only` then return early and just return the distance matrix. This is 
	useful for doing thresholding. 
	"""
    ## compute the distance matrix between the inf proteins and the 1311 OpenCell proteins
    if dist is None:
        dist = pairwise_distances(embeds_consensus_inf,
                                  embeds_consensus_opencell,
                                  metric=mode)
        dist = torch.from_numpy(dist)

    # matrix (n_well_ids_inf, n_prots_opencell), which here is (80,1311)
    dist_argsort = torch.argsort(dist, axis=1)[:, :k_nearest_prots]
    prots_nearest = df_opencell_lookup['prot_id'].values[
        dist_argsort]  # (n_well_ids_inf,k_nearest_prots)
    dist_nearest = np.zeros(dist_argsort.shape)
    for i in range(len(dist_nearest)):
        dist_nearest[i] = dist[i, dist_argsort[i]]
    if mode == 'correlation':
        dist_nearest = 1 - dist_nearest
    loc_grade1_nearest = df_opencell_lookup['loc_grade1'].values[
        dist_argsort]  # (n_well_ids_inf,k_nearest_prots)

    if return_dists_only:
        return dist, dist_argsort, prots_nearest, loc_grade1_nearest

    ## stick the knn preds together. Note that the rows of `df_inf_lookup` are aligned
    # with the rows of `prots_nearest` and `loc_grade1_nearest`. This was checked in
    # the function caller (it should check that `df_inf_lookup` algin with `labels_consensus_inf`
    df_results_summary = df_inf_lookup[[
        'well_id', 'protein', 'orgIP_cluster_annotation',
        'plate33_eyeballed_annotation'
    ]]
    df_results_summary.index = range(len(df_results_summary))
    df_prot_nearest = pd.DataFrame(
        prots_nearest,
        columns=[f'prot_nn{i}' for i in range(prots_nearest.shape[1])])
    df_dist_nearest = pd.DataFrame(
        dist_nearest,
        columns=[f'dist{i}' for i in range(prots_nearest.shape[1])])
    df_loc_grade1_nearest = pd.DataFrame(
        loc_grade1_nearest,
        columns=[
            f'loc_grade1_nn{i}' for i in range(loc_grade1_nearest.shape[1])
        ])

    df_results_summary = pd.concat([
        df_results_summary,
        df_prot_nearest,
        df_dist_nearest,
        df_loc_grade1_nearest,
    ],
                                   axis=1)
    # measure the loss for the nearest neighbour proteins
    if do_include_nn_losses:
        m, n = prots_nearest.shape
        loss_ce = np.zeros((m, n))
        loss_recon = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                loss_ce[i, j] = lookup_loss_ce[prots_nearest[i, j]]
                loss_recon[i, j] = lookup_loss_recon[prots_nearest[i, j]]

        df_loss_ce = pd.DataFrame(
            loss_ce,
            columns=[f'loss_ce{i}' for i in range(prots_nearest.shape[1])])
        df_loss_recon = pd.DataFrame(
            loss_recon,
            columns=[f'loss_recon{i}' for i in range(prots_nearest.shape[1])])

        df_results_summary = pd.concat(
            [df_results_summary, df_loss_ce, df_loss_recon], axis=1)

    # save the 'knn predictions'
    f_save = dir_results / f"knn_opencell_preds_{fname_stem}.csv"
    print(f"saving results csv [{f_save}]")
    df_results_summary.transpose().to_csv(f_save)

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
                   fname_stem,
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
        imgs.append(crops_inf[idxs_img[:n_samples]][:, :2])

        # get the nearest protein set
        prots_nearest = []
        dist_nearest = []
        for j in range(k_nearest_prots):
            prot = row[f'prot_nn{j}']
            prots_nearest.append(prot)
            idxs_img = np.where(df_opencell['prot_id'] == prot)[0]
            np.random.shuffle(idxs_img)
            imgs.append(torch.from_numpy(crops_opencell[idxs_img[:n_samples]]))

            dist_ = row[f'dist{j}']
            dist_nearest.append(f"{dist_:.2f}")

        imgs = torch.cat(imgs)
        imgs = norm_0_1_perimg_perch(imgs)
        imgs_show = torch.zeros(len(imgs),
                                3,
                                *imgs.shape[-2:],
                                dtype=imgs.dtype)
        imgs_show[:, 1] = imgs[:, 0]  # protein in green channel
        imgs_show[:, 2] = imgs[:, 1]  # nucleus in green channel
        grid = make_grid(imgs_show, nrow=n_samples).permute(1, 2, 0)
        dir_viz_ = dir_viz / ('viz_' + fname_stem)
        dir_viz_.mkdir(exist_ok=True)
        f_save = dir_viz_ / f"knn_samples_prot_{protein}_wellid_{well_id}.png"

        title = f"well_{well_id}_prot_{protein}\nnearest={prots_nearest}\ndist={dist_nearest}"
        f, axs = plt.subplots(figsize=(15, 15))
        axs.imshow(grid)
        axs.set(title=title)
        f.savefig(f_save, dpi=200)
        plt.close()

    pass


def get_consensus_encoding(embed_data,
                           labels,
                           sort_key=None,
                           do_filtering=False,
                           crops=None):
    """  
	`crops` only used for DEBUG=True
	"""
    labels_uniq = np.array(sorted(np.unique(labels), key=sort_key))  # (1311,)

    embeds_consensus = []

    assert embed_data.shape[0] in (
        1, 8
    ), "embed_data should be shape (n_augs,n_samples,ndim), where n_augs is 1 or 8 for rotation augmentation"

    for label in labels_uniq:
        idxs = np.where(labels == label)[0]

        if do_filtering and label in protein_filter.keys():
            idxs = idxs[protein_filter[label]]

            DEBUG = 0
            if label == "GLT8D1":
                print(label, len(idxs))
            if DEBUG:
                imgs = make_crops_imgs(crops[idxs])
                grid = make_grid(imgs, ncols=10)
                f, axs = plt.subplots()
                axs.imshow(grid.permute(1, 2, 0))
                axs.set_axis_off()
                f.savefig("tmp.png", dpi=200)

        embed = embed_data[:, idxs]
        embed = embed.view(-1, embed.shape[-1]).mean(0)
        embeds_consensus.append(embed)
    embeds_consensus = torch.vstack(embeds_consensus)

    return embeds_consensus, labels_uniq


def threshold_based_filtering(self):
    raise NotImplementedError(
        "below is the code that did this, but it needs to be rewritten to be usable"
    )
    # prots_nearest = df_opencell_lookup['prot_id'].values[dist_argsort] # (n_well_ids_inf,k_nearest_prots)
    # loc_grade1_nearest = df_opencell_lookup['loc_grade1'].values[dist_argsort] # (n_well_ids_inf,k_nearest_prots)
    threshold = 0.6
    is_under = dist < threshold
    is_under = is_under.fill_diagonal_(0)
    n_matches = is_under.sum(1)

    matches_loc_grade1_all = []
    votes_all = []
    prediction_all = []
    df_inf_predictions = df_inf_lookup.copy()
    for i in range(n_inf):
        row_index = df_inf_predictions.iloc[i].name
        matches_loc_grade1 = df_opencell_lookup['loc_grade1'].values[
            is_under[i]]
        matches_loc_grade1_all.append(matches_loc_grade1)
        res = order_by_frequency_with_counts(matches_loc_grade1)
        votes_all.append(res)
        prediction_all.append(res[0][0])

        # df_inf_predictions['pred_votes'].iloc[row_index] = res
        # df_inf_predictions['top_pred'].iloc[row_index ] = res[0][0]
        # print( df_inf_lookup[['protein','orgIP_cluster_annotation','plate33_eyeballed_annotation']].iloc[i])
        # print(res)
        # print()
        # print()

    df_inf_predictions.to_csv(f_save)
    df_inf_predictions['predictions'] = prediction_all
    df_inf_predictions['votes'] = votes_all
    f_save = dir_results / f"threshold_opencell_preds_mode_{mode}_threshold_{threshold:.03f}.csv"
    df_inf_predictions.to_csv(f_save)


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


def analysis_choose_opencell_threshold(dir_pretrained_model,
                                       mode="correlation",
                                       do_umap=False):
    current_filename = Path(os.path.basename(__file__))
    dir_results = Path(
        "inference/results") / current_filename.stem / dir_pretrained_model
    dir_results.mkdir(exist_ok=True, parents=True)

    # results dirsgg
    embed_opencell, labels_opencell, crops_opencell, df_opencell, df_opencell_lookup = get_embeds_and_crops(
        dir_pretrained_model,
        checkpoint=checkpoint,
        representation=representation,
        dataset="opencell")

    ## get "consensus encodings" - this one is the elementwise mean of the vqvec2 vecotors
    protein_label_opencell = labels_opencell[:, 1]
    # get 'consensus encoding' which is one vector for the whole protein
    embeds_consensus_opencell, labels_consensus_opencell = get_consensus_encoding(
        embed_opencell, protein_label_opencell)

    #
    if do_umap:
        print("Doing umap embeddings")
        import umap
        reducer = umap.UMAP(n_components=2, verbose=True)
        embedding = reducer.fit_transform(embeds_consensus_opencell)
        # overwrite it
        embeds_consensus_opencell = embedding

    # measure distances
    if mode == "correlation":
        dist = pairwise_distances(embeds_consensus_opencell,
                                  embeds_consensus_opencell,
                                  metric='correlation')
    elif mode == "euclidean":
        dist = pairwise_distances(embeds_consensus_opencell,
                                  embeds_consensus_opencell,
                                  metric='euclidean')
    else:
        raise
    dist = torch.from_numpy(dist)

    ## let's plot the distribution of distances of one protein to other proteins, grouped
    ## by whether or not they have the same loc_grade1 annotation.
    is_same_all_classes = np.zeros_like(dist, dtype=np.int8)
    is_overlapping_classes = np.zeros_like(dist, dtype=np.int8)
    loc_grade_1 = np.array(df_opencell_lookup['loc_grade1'])

    # (this bit should be solvable with broadcasting, but it's too annoying)
    n = len(dist)
    for i in range(n):
        for j in range(n):
            is_same_all_classes[i, j] = (loc_grade_1[i] == loc_grade_1[j])
            overlapping_classes = set(loc_grade_1[i].split(";")).intersection(
                set(loc_grade_1[j].split(";")))
            is_overlapping_classes[i, j] = (len(overlapping_classes) > 0)
    is_no_overlapping_classes = -1 * (is_overlapping_classes - 1)
    np.fill_diagonal(is_same_all_classes, 0)
    np.fill_diagonal(is_overlapping_classes, 0)
    np.fill_diagonal(is_no_overlapping_classes, 0)

    dists_same_all_classes = dist[np.where(is_same_all_classes)]
    dists_overlapping_classes = dist[np.where(is_overlapping_classes)]
    dists_no_overlap = dist[np.where(is_no_overlapping_classes)]

    import matplotlib.pyplot as plt
    f, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 20))
    bins = 100
    axs[0].hist(dists_same_all_classes, bins=bins)
    axs[1].hist(dists_overlapping_classes, bins=bins)
    axs[2].hist(dists_no_overlap, bins=bins)

    f_save = dir_results / f"compare_distances_1_mode_{mode}_doUmap_{do_umap}.png"
    f.savefig(f_save)
    plt.close()

    ### let's do the same thing but on a per-class basis (where class is defined as each specific combo, ~100 classes)
    uniq_loc_grade_1 = np.unique(loc_grade_1)
    f, axs = plt.subplots(3,
                          len(uniq_loc_grade_1),
                          figsize=(500, 30),
                          sharey=True)
    if 0:  # there's a bug here, no time to deal with it
        for i, cls_loc_grade_1 in tqdm(enumerate(uniq_loc_grade_1)):
            # index out prots from only that class. Need to take the subset of both
            # the distance matrix and the masking matrices

            idxs = np.where(loc_grade_1 == cls_loc_grade_1)[0]
            dist_subset = dist[idxs][:, idxs]
            is_same_all_classes_subset = is_same_all_classes[idxs][:, idxs]
            is_overlapping_classes_subset = is_overlapping_classes[idxs][:,
                                                                         idxs]
            is_no_overlapping_classes_subset = is_no_overlapping_classes[
                idxs][:, idxs]

            dists_same_all_classes_subset = dist_subset[np.where(
                is_same_all_classes_subset)]
            dists_overlapping_classes_subset = dist_subset[np.where(
                is_overlapping_classes_subset)]
            dists_no_overlap_subset = dist_subset[np.where(
                is_no_overlapping_classes_subset)]

            try:
                axs[0, i].set(title=f"n={len(idxs)} - {cls_loc_grade_1}")
                axs[0, i].hist(dists_same_all_classes_subset, bins=bins)
                axs[1, i].hist(dists_overlapping_classes_subset, bins=bins)
                axs[2, i].hist(dists_no_overlap_subset, bins=bins)
            except:
                print(
                    f"Could not make histogram for class {cls_loc_grade_1}, n={len(cls_loc_grade_1)}"
                )

        f_save = dir_results / f"compare_distances_1_all_classes_mode_{mode}_doUmap_{do_umap}.png"
        f.savefig(f_save)

    ## now let's suppose that we chose some threshold `t`. How many other proteins
    ## would fit within that threshold? For each protein, we cound them, `k`, and
    ## now let's plot a histogram of `k`. The idea: if there's lots that are 0,
    ## then it's too tight. Plot it for a range of `t`.

    ts = np.linspace(0, 0.6, 10)

    f, axs = plt.subplots(len(ts), 1, figsize=(5, 10), sharex=True)
    bins = 50
    for i, t in tqdm(enumerate(ts)):
        is_under = np.array(dist < t).astype(int)
        np.fill_diagonal(is_under, 0)
        cnt_under_t = is_under.sum(
            1)  # for each prot, how many other prots under threshold
        axs[i].hist(cnt_under_t, bins=bins)
        axs[i].set(title=f"threshod {t:.3f}")

    f_save = dir_results / f"simulate_threshold_buckets_1_mode_{mode}_doUmap_{do_umap}.png"
    f.savefig(f_save)

    pass


def make_crops_imgs(imgs):
    """ 
	image crops are either 2 channel (protein+nucleus) or 3-channel (protein+
	nucleus+distance transform). Make an image where the protein is in the green
	channel, and the nucleus in the blue channel, and the red channel is all 0s.
	"""
    imgs_show = torch.zeros(len(imgs), 3, *imgs.shape[-2:], dtype=imgs.dtype)
    imgs_show[:, 1] = imgs[:, 0]  # protein in green channel
    imgs_show[:, 2] = imgs[:, 1]  # nucleus in green channel

    return imgs_show


def generate_inference_crop_grids(dir_pretrained_model,
                                  fname_crops,
                                  nsamples=100,
                                  ncols=10,
                                  checkpoint=None):
    """
	"""
    dir_results = Path(
        "inference/results") / current_filename.stem / dir_pretrained_model
    dir_results.mkdir(exist_ok=True, parents=True)
    dir_viz = dir_results / 'viz_proteins'
    dir_viz.mkdir(exist_ok=True, parents=True)

    # get the images and metadata (and also embeddings as a byproduct)
    embed_inf, labels_inf, crops_inf, df_meta_inf, df_annotations, df_inf_lookup = get_embeds_and_crops(
        dir_pretrained_model,
        checkpoint=checkpoint,
        representation='vqvec2',
        dataset='inference')

    for protein in df_meta_inf['protein'].unique():
        # protein = "GLT8D1"
        idxs = np.where(df_meta_inf['protein'] == protein)[0][:nsamples]
        imgs = crops_inf[idxs]
        imgs_show = torch.zeros(len(imgs),
                                3,
                                *imgs.shape[-2:],
                                dtype=imgs.dtype)
        imgs_show[:, 1] = imgs[:, 0]  # protein in green channel
        imgs_show[:, 2] = imgs[:, 1]  # nucleus in green channel
        grid = make_grid(imgs_show, nrow=ncols).permute(1, 2, 0)

        f_save = dir_viz / f"viz_{protein}.png"
        f, axs = plt.subplots()
        axs.imshow(grid)
        axs.set_axis_off()
        f.savefig(f_save, dpi=500)

    pass

    # grid = make_grid()


if __name__ == "__main__":
    fname_crops = "inference/results/crop/crops.pt"
    fname_crops_meta = "inference/results/crop/crops_meta.csv"
    # this file is protein ids and estimates of their annotations
    fname_annotations = "data/cz_infectedcell_finalwellmapping.csv"
    pca_dim = 200

    # generate_inference_crop_grids("results/20231218_train_all_no_nucdist_balanced_classes_2",
    # fname_crops, nsamples=100, ncols=10)

    dir_pretrained_models_checkpoints = [
        # ["results/20231218_train_all_balanced_classes_2", None],
        ["results/20231222_train_all_balanced_classes_1", None],
        ["results/20231218_train_all_no_nucdist_balanced_classes_2", None],
        ["results/20231222_train_all_no_nucdist_balanced_classes_1", None],
        ["results/20231222_train_all_balanced_classes_1", None],
        ["results/20231221_train_with_orphans", None],
        ["results/20231221_train_with_orphans_no_nucdist", None],
        ["results/20231222_train_all_no_nucdist_balanced_classes_1", None],
        ["results/20231222_train_all_balanced_classes_1", None],
        ["results/20231218_train_all_no_nucdist", None],
        ["results/20231022_train_all", None],
    ]

    # analyse distances
    representation = 'vqvec2'  #'hist'
    mode = 'correlation'
    combine_inference_wells = True
    do_include_nn_losses = True
    # do_filtering = True

    do_visualize_knns = True  # create an image for each protein visualizing nearest-neighbors
    combine_augmented_embeddings = True

    # for representation in ('vqvec2', 'hist'):
    for (dir_pretrained_model,
         checkpoint) in dir_pretrained_models_checkpoints:
        for representation in ('vqvec2', 'hist'):
            for pca_dim in (
                    200,
                    None,
            ):
                for mode in (
                        'correlation',
                        'euclidean',
                ):
                    for do_filtering in (
                            # False,
                            True, ):
                        for combine_augmented_embeddings in (
                                # False,
                                True, ):
                            get_nearest_proteins(
                                fname_crops=fname_crops,
                                fname_crops_meta=fname_crops_meta,
                                fname_annotations=fname_annotations,
                                dir_pretrained_model=dir_pretrained_model,
                                representation=representation,
                                combine_inference_wells=combine_inference_wells,
                                pca_dim=pca_dim,
                                do_filtering=do_filtering,
                                do_visualize_knns=do_visualize_knns,
                                checkpoint=checkpoint,
                                combine_augmented_embeddings=
                                combine_augmented_embeddings,
                                do_include_nn_losses=do_include_nn_losses,
                                mode=mode)

    ipdb.set_trace()
