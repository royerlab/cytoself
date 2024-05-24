"""
Run in project home dir as 
	python inference/get_crop_features.py

Get the crops from `inference/results/crops/crops.pt` and metadata from 
`inference/results/crops/crops_meta.csv`. Create a `datamanager` object (an 
object type that was defined for the cytoself project). At this stage 

Following the code in `example_scripts/simple_example.py`, 

Embeddings will put in `inference/results/get_crop_features/<pretrained_model_dir>`
where `<pretrained_model_dir>` is an arg to `get_features()`. This way, we can 
do the analysis for multiple pretrained models.
"""
import ipdb
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import json
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances
from skimage import io
from skimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import distance_transform_edt
import tqdm
import pandas as pd

from cytoself.datamanager.preloaded_dataset import PreloadedDataset
from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer

# description of the versions in the 
VERSION = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_features(fname_crops,
				 fname_crops_meta,
				 dir_pretrained_model,
				 checkpoint=None,
				 do_reconstructions=False,
				 augmentation=None,
				 do_retrieval_test=False,
				 ):
	"""
	augmentation: one of 8 image augmentations. None,90,180,270,0_f,90_f,180_f,270_f
	This will rotate/flip the image 
	and save the embeddgin to a different file with the augmentation as the suffix
	(unless augmentation=None, then there is no suffix)

	"""
	# handle image augmentations 
	stem_fname = augmentation
	if augmentation is None: 
		augmentation = "0"
	assert augmentation in ("0","90","180","270","0_f","90_f","180_f","270_f")
	augmentation_rot = int(augmentation.split("_")[0])
	augmentation_do_flip = True if len(augmentation.split("_"))>1 else False

	# results dirs
	current_filename = Path(os.path.basename(__file__))
	dir_results = Path(
		"inference/results") / current_filename.stem / dir_pretrained_model / f"ckpt_{checkpoint}"
	if VERSION > 0:
		dir_results = dir_results / f"version{VERSION}"
	dir_results.mkdir(exist_ok=True, parents=True)
	dir_viz = Path("inference/viz") / current_filename.stem
	dir_viz.mkdir(exist_ok=True, parents=True)

	# reproduce the trainer object
	model_args, train_args = get_model_trainer_args(dir_pretrained_model)
	trainer = CytoselfFullTrainer(train_args,
								  homepath=dir_results,
								  model_args=model_args)

	# the model was created by the trainer.
	# Load the 'best' checkpoint
	if checkpoint is None:
		fs_models = [f for f in Path(dir_pretrained_model).glob("model*pt")]
		assert len(fs_models) == 1
		fs_model = fs_models[0]
		trainer.model = torch.load(fs_model)

	else:
		fs_model = Path(dir_pretrained_model
						) / f'checkpoints/checkpoint_ep{checkpoint}.chkp'
		chkp = torch.load(fs_model)
		trainer.model.load_state_dict(chkp['model_state_dict'])

	# load crops and metadata. Put it into datamanager.
	crops = torch.load(fname_crops)
	assert crops.shape[-2:] == (200, 200)
	crops = T.Resize(100)(crops).to(device)
	if augmentation_rot != 0:
		crops = T.functional.rotate(crops, augmentation_rot)
	if augmentation_do_flip:
		crops = T.functional.vflip(crops)

	if model_args['input_shape'][0] == 2:
		crops = crops[:, :2]

	## now deal with augmentations 
	df_meta = pd.read_csv(fname_crops_meta)
	labels, map_int_to_wellid = get_labels_from_df_meta(df_meta)
	datamanager = PreloadedDataset(label=labels, data=crops)

	# validation: image reconstructions - results in`inference/viz/get_crop_features/reconstructions`
	if do_reconstructions:
		test_reconstructions(trainer, crops, dir_viz, n_samples=20)

	# the standard embeddings: the output of the vq encoder at level 2
	loader_inf = DataLoader(datamanager, batch_size=64)
	trainer.model.eval()
	embeddings_vqvec2 = trainer.infer_embeddings(loader_inf,
												 output_layer='vqvec2')
	f_save = dir_results / f"embeddings_vqvec2_aug{stem_fname}.pt"
	torch.save(embeddings_vqvec2, f_save); 

	# the feature spectrum encodings
	vqindhist1 = trainer.infer_embeddings(loader_inf, 'vqindhist1')
	f_save = dir_results / f"embeddings_vqindhist1_aug{stem_fname}.pt"
	torch.save(vqindhist1, f_save)

	# re-save the metadata again
	df_meta.to_csv(dir_results / (Path(fname_crops_meta).stem + ".csv"))

	# another validation check: knn lookup

	if do_retrieval_test:
		test_knn_retrieval_embeddings(crops,
									  embeddings_vqvec2,
									  dir_viz=dir_viz,
									  metric='euclidean')
		test_knn_retrieval_embeddings(crops,
									  embeddings_vqvec2,
									  dir_viz=dir_viz,
									  metric='correlation')


def get_labels_from_df_meta(df_meta):
	""" 
	Take df_meta dataframe, and convert to a tensor of integer labels, where 
	each well_id is its own class. 

	Output:
		labels (np.array): shape (n,3). Col0 is an int for well_id, col1 is 
			fov_id, and col2 is a unique identifier for the data sample.
		map_int_to_wellid (dict): dictionary mapping ints in labels[:,0] to a 
			well_id like "A10""
	"""
	y = np.array([Path(l).stem.split("_")[2:4] for l in df_meta.fname_pro])
	well_ids = y[:, 0]
	fov_ids = y[:, 1].astype(int)
	well_ids_uniq, fov_ids_uniq = np.unique(well_ids), np.unique(fov_ids)

	map_int_to_wellid = dict(zip(range(len(well_ids_uniq)), well_ids_uniq))
	map_wellid_to_int = {v: k for k, v in map_int_to_wellid.items()}
	well_ids_int = np.array([map_wellid_to_int[w] for w in well_ids])

	labels = np.stack((well_ids_int, fov_ids, np.arange(len(well_ids_int))),
					  axis=1)

	return labels, map_int_to_wellid


def get_model_trainer_args(dir_pretrained_model):
	dir_pretrained_model = Path(dir_pretrained_model)

	f_args = dir_pretrained_model / "model_args.json"
	assert f_args.exists(), f_args
	with open(f_args, 'r') as f:
		model_args = json.load(f)
	f_args = dir_pretrained_model / "train_args.json"
	assert f_args.exists(), f_args
	with open(dir_pretrained_model / "train_args.json", 'r') as f:
		train_args = json.load(f)
	return model_args, train_args


def test_reconstructions(trainer, crops, dir_viz, n_samples=20, dpi=300):

	dir_viz_recons = Path(dir_viz) / "test_reconstructions"
	dir_viz_recons.mkdir(exist_ok=True, parents=True)

	# randomly sample some of the images
	dtype = crops.dtype
	idxs = np.arange(len(crops))
	np.random.seed(0)
	np.random.shuffle(idxs)
	idxs = idxs[:n_samples]

	# reconstruction with autoencoder
	crops_test = crops[idxs].numpy()
	crops_recon = trainer.infer_reconstruction(crops_test)

	# visualize
	imgs = torch.zeros((len(crops_test) * 2, *crops_test.shape[1:]),
					   dtype=dtype)
	imgs[::2] = torch.tensor(crops_test)
	imgs[1::2] = torch.tensor(crops_recon)
	grid_pro = make_grid(imgs, nrow=10)[0]
	grid_nuc = make_grid(imgs, nrow=10)[1]
	f, axs = plt.subplots()
	axs.imshow(grid_pro)
	axs.set_axis_off()
	f.savefig(dir_viz_recons / "reconstructions_protein.png", dpi=dpi)
	f, axs = plt.subplots()
	axs.imshow(grid_nuc)
	axs.set_axis_off()
	f.savefig(dir_viz_recons / "reconstructions_nucleus.png", dpi=dpi)


def test_knn_retrieval_embeddings(crops,
								  embeddings_vqvec2,
								  dir_viz,
								  metric='euclidean',
								  n_samples=30,
								  k_nearest=20,
								  seed=0,
								  dpi=200):
	""" 
	Get the nearest `k` images from the same dataset (includeing itself) for `n`
	images.
	The `metric` is passed to sklearn.metrics.pairwise_distances. 
	n_samples (int): how many images to do retrieval for
	seed (int): for sampling query images in the knn retrieval.
	"""
	dir_viz_knn = Path(dir_viz) / f"test_knn"
	dir_viz_knn.mkdir(exist_ok=True, parents=True)

	# get the distrance matrix
	z = torch.tensor(embeddings_vqvec2[0])
	z = z.view(len(z), -1)
	dist = pairwise_distances(z, metric=metric)

	# randomly sample some of the images
	idxs = np.arange(len(z))
	np.random.seed(seed)
	np.random.shuffle(idxs)
	idxs = idxs[:n_samples]

	# for the image subset, to knn retrieval
	dist_argsort = torch.argsort(torch.from_numpy(dist[idxs]), dim=1)
	idxs_nearest = dist_argsort[:, :k_nearest]  # get the k_nearest indexes
	idxs_nearest = idxs_nearest.flatten()
	imgs = crops[idxs_nearest]
	grid = make_grid(imgs, nrow=k_nearest)
	grid_pro = grid[0]
	grid_nuc = grid[1]

	# visualize
	f, axs = plt.subplots()
	axs.imshow(grid_pro)
	axs.set_axis_off()
	f.savefig(dir_viz_knn / f"knn_{metric}_seed{seed}_protein.png", dpi=dpi)
	f, axs = plt.subplots()
	axs.imshow(grid_nuc)
	axs.set_axis_off()
	f.savefig(dir_viz_knn / f"knn_{metric}_seed{seed}_nucleus.png", dpi=dpi)


def get_features_opencell(fname_crops,
						  fname_crops_meta,
						  dir_pretrained_model,
						  checkpoint=None,
						  augmentation=None,
						  do_retrieval_test=False,
						  do_reconstructions=False):

	# handle image augmentations 
	stem_fname = augmentation
	if augmentation is None: 
		augmentation = "0"
	assert augmentation in ("0","90","180","270","0_f","90_f","180_f","270_f")
	augmentation_rot = int(augmentation.split("_")[0])
	augmentation_do_flip = True if len(augmentation.split("_"))>1 else False

	# results dirs
	current_filename = Path(os.path.basename(__file__))
	dir_results = Path(
		"inference/results") / current_filename.stem / dir_pretrained_model / f"ckpt_{checkpoint}"
	dir_results.mkdir(exist_ok=True, parents=True)

	dir_viz = Path("inference/viz") / current_filename.stem
	dir_viz.mkdir(exist_ok=True, parents=True)

	# reproduce the trainer object
	model_args, train_args = get_model_trainer_args(dir_pretrained_model)
	trainer = CytoselfFullTrainer(train_args,
								  homepath=dir_results,
								  model_args=model_args)

	# the model was created by the trainer.
	# Load the 'best' checkpoint
	if checkpoint is None:
		fs_models = [f for f in Path(dir_pretrained_model).glob("model*pt")]
		assert len(fs_models) == 1
		fs_model = fs_models[0]
		trainer.model = torch.load(fs_model)

	else:
		fs_model = Path(dir_pretrained_model
						) / f'checkpoints/checkpoint_ep{checkpoint}.chkp'
		chkp = torch.load(fs_model)
		trainer.model.load_state_dict(chkp['model_state_dict'])

	# load crops and metadata. Put it into datamanager.
	crops = np.load(fname_crops)
	# assert crops.shape[-2:] == (200,200)
	# crops = T.Resize(100)(crops)
	crops = torch.from_numpy(crops)
	if augmentation_rot != 0:
		crops = T.functional.rotate(crops, augmentation_rot)
	if augmentation_do_flip:
		crops = T.functional.vflip(crops)

	if model_args['input_shape'][0] == 2:
		crops = crops[:, :2]

	if model_args['input_shape'][0] == 2:
		crops = crops[:, :2]

	# df_meta = pd.read_csv(fname_crops_meta)
	df_meta = np.load(fname_crops_meta)
	labels = np.arange(len(crops))[:, None]
	datamanager = PreloadedDataset(label=labels, data=crops)

	# validation: image reconstructions - results in`inference/viz/get_crop_features/reconstructions`
	if do_reconstructions:
		test_reconstructions(trainer, crops, dir_viz, n_samples=20)

	# the standard embeddings: the output of the vq encoder at level 2
	loader_inf = DataLoader(datamanager, batch_size=64)
	trainer.model.eval()
	embeddings_vqvec2 = trainer.infer_embeddings(loader_inf,
												 output_layer='vqvec2')
	f_save = dir_results / f"embeddings_vqvec2_opencell_aug{stem_fname}.pt"
	torch.save(embeddings_vqvec2, f_save)  #

	# the feature spectrum encodings
	vqindhist1 = trainer.infer_embeddings(loader_inf, 'vqindhist1')
	f_save = dir_results / f"embeddings_vqindhist1_opencell_aug{stem_fname}.pt"
	torch.save(vqindhist1, f_save)

	loss_ce_per_protein, loss_recon_per_protein = get_protein_losses(trainer, crops, df_meta, dir_pretrained_model)
	loss_ce_per_protein.to_csv(dir_results / "loss_ce.csv")
	loss_recon_per_protein.to_csv(dir_results / "loss_recon.csv")

def get_protein_losses(trainer, crops, df_meta, dir_pretrained_model):
	""" """
	from torch.utils.data import DataLoader, TensorDataset
	uniq_prot_locgrade = np.unique(df_meta[:,[1,2]], axis=0)
	lookup_prot_to_locgrade = dict(zip(uniq_prot_locgrade[:,0], uniq_prot_locgrade[:,1]))

	# the classification indices are defined by the ensg_id
	ensg_id = df_meta[:,0]
	ensg_id_uniq = np.unique(ensg_id)
	lookup_ensg_id_to_label = dict(zip(ensg_id_uniq, np.arange(len(ensg_id_uniq))))
	labels = np.array([lookup_ensg_id_to_label[i] for i in ensg_id])
	# also get the protein names
	proteins = df_meta[:,1]
	proteins_uniq = np.sort(np.unique(proteins))
	lookup_prot_to_label = dict(zip(proteins_uniq, np.arange(len(proteins_uniq))))
	lookup_label_to_prot = {v:k for k,v in lookup_prot_to_label.items()}

	dset = TensorDataset(crops, torch.from_numpy(labels))
	loader = DataLoader(dset, batch_size=512)
	crops_pred, logits = [], [] 
	with torch.no_grad():
		for img, label in tqdm.tqdm(loader):
			img = img.to(trainer.device)
			crops_pred_, logits_ = trainer.model(img) 
			crops_pred.append(crops_pred_)
			logits.append(logits_)

		crops_pred = torch.cat(crops_pred).cpu()
		logits = torch.cat(logits).cpu()

	# careful: not sure if the datamanager does normalization of the images
	loss_recon = torch.nn.functional.mse_loss(crops_pred, crops, reduction="none")
	loss_recon = loss_recon.view(len(loss_recon), -1).mean(1)

	# loss_ce
	loss_ce = torch.nn.functional.cross_entropy(logits, torch.from_numpy(labels), reduction="none")
	preds = torch.argmax(logits, dim=1)
	preds_protein = np.array([lookup_label_to_prot.get(i.item(),"") for i in preds])

	# analye the reconstruction losses 
	df_loss_recon = pd.DataFrame(dict(loss_recon=loss_recon, protein=proteins))
	loss_recon_per_protein = df_loss_recon.groupby("protein")['loss_recon'].mean()
	prots_sorted = loss_recon_per_protein.sort_values().index.values[::-1]

	# analyse the ce loss 
	df_loss_ce = pd.DataFrame(dict(loss_ce=loss_ce, protein=proteins))
	loss_ce_per_protein = df_loss_ce.groupby("protein")['loss_ce'].mean()

	idxs_least = np.argsort(loss_ce_per_protein.values)[::-1]
	prots_sorted = loss_ce_per_protein.sort_values().index.values[::-1]

	if 0:
		import matplotlib.pyplot as plt
		f, axs = plt.subplots()
		axs.hist(loss_ce_per_protein, bins=50)
		f.savefig("hist_2.png")

		# plot the cumulative density 
		f, axs = plt.subplots()
		data_sorted = np.sort(loss_ce_per_protein)
		cumulative = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
		axs.plot(data_sorted, cumulative, marker='.', linestyle='none')
		f.savefig("cdf_2.png")

	return loss_ce_per_protein, loss_recon_per_protein



if __name__ == "__main__":

	# a tuple with the pretrained directory and the chosen checkpoint (None if best)
	dir_pretrained_models_checkpoints = [

		# ["results/20231218_train_all_balanced_classes_2", None],
		# ["results/20231218_train_all_no_nucdist_balanced_classes_2", None],    
		# ["results/20231222_train_all_no_nucdist_balanced_classes_1", None],
		# ["results/20231222_train_all_balanced_classes_1", None],
		# ["results/20231221_train_with_orphans", None], 
		# ["results/20231221_train_with_orphans_no_nucdist", None],
		# ["results/20231222_train_all_no_nucdist_balanced_classes_1", None],
		# ["results/20231222_train_all_balanced_classes_1", None],
		# ["results/20231218_train_all_no_nucdist", None],
		# ["results/20231022_train_all", None],
		["results/20240129_train_all_no_nucdist_balanced_classes_1", 36],
		["results/20240129_train_all_balanced_classes_2", None],
		["results/20240129_train_all", None],
		["results/20240129_train_all_no_nucdist", None],
	]
	# inference crops
	if VERSION > 0 :
		fname_crops = f"inference/results/crop/crops_v{VERSION}.pt"
		fname_crops_meta = f"inference/results/crop/crops_meta_v{VERSION}.csv"
	else:
		fname_crops = f"inference/results/crop/crops.pt"
		fname_crops_meta = "inference/results/crop/crops_meta.csv"
	do_reconstructions = False 

	for augmentation in ("0","90","180","270","0_f","90_f","180_f","270_f"):
		for (dir_pretrained_model, checkpoint) in dir_pretrained_models_checkpoints:
		# for augmentation in ("0"): # (,"90","180","270","0_f","90_f","180_f","270_f"):
			get_features(
				fname_crops=fname_crops,
				fname_crops_meta=fname_crops_meta,
				dir_pretrained_model=dir_pretrained_model,
				checkpoint=checkpoint,
				do_reconstructions=do_reconstructions,
				augmentation=augmentation,
				do_retrieval_test=False,
			)

			# get the features from OpenCell TODO: consolidate into one function

			if "2023" in dir_pretrained_model:
			   dir_opencell_test = Path("data/test_dataset_metadata")
			elif "2024" in dir_pretrained_model:
				dir_opencell_test = Path("data/test_dataset_metadata_v1/data/opencell_crops_processed2/")
			else:
				raise 

			# fname_crops_opencell = dir_opencell_test / "test_dataset_crops.npy"
			# fname_crops_opencell_meta = dir_opencell_test / "test_dataset_labels.npy"

			# get_features_opencell(
			# 	fname_crops=fname_crops_opencell,
			# 	fname_crops_meta=fname_crops_opencell_meta,
			# 	dir_pretrained_model=dir_pretrained_model,
			# 	checkpoint=checkpoint,
			# 	do_reconstructions=do_reconstructions,
			# 	augmentation=augmentation,
			# 	do_retrieval_test=False,
			# )

candidate_shitty_proteins = ["MYO19","DNAJC11"]
candidate_shitty_proteins = ["MYO19","MFN2","VDAC2","DNAJC11","VDAC3"] # mito 
candidate_shitty_proteins = ["SLC37A3","PI4K2A"]

candidate_good_proteins = ["STT3A","CERS6","C4orf32","CERS5",]


