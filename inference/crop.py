"""
Run in project home dir as 
	python inference/crop.py

Get the FOVs for `inference/results/load_inf_data/` and the segmentation maps 
from `inference/results/nucelear_segmentation/all_segmasks.pt`.

For each nucleus, generate a crop centered around that nucleus with 200px width
and height. If the crop extends beyond the boundaries of the FOV image (i.e. it's
too close to the border), then we skip it.
The crop has the protein in ch0 and protein in ch1. They are saved to 
`inference/results/crop/`. Each channel in each crop is normalized to range [0,1]
independently to the rest of the FOV. 

The output is `inference/results/crops/crops.pt`

Also visualize the crops in, inference/viz/crop/. Create one image in that folder
for nucleus and one for protein. Each grid item is a crop around one nucleus.

Optionally add a filter for whether the protein channel has enough signal. 

** I really should have put the two separate functions into one function, but what can you do
"""

from pathlib import Path
import ipdb
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from PIL import UnidentifiedImageError
from skimage import measure
from torchvision.utils import make_grid
import os
from skimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import distance_transform_edt
import torchvision.transforms as T
from aicsimageio import AICSImage
import tqdm

# VERSION defines the cropping strategy, and defines how the crops will be saved 
# If 0 do the normalization on each crop (recommended)
# If 1, do the normalization on each FOV. 
# If 2, do crop-level normalization (just like version 0) AND also do intensity-threshold based filtering. 
# 		This version produces logs btw. 
VERSION = 0

# these params are all relevant for VERSION 2 only 
LOG_VERSION_2_RESULTS = 1 
THRESH = 10
PCNT_PIXELS = 0.01
num_pixels_over_thresh = -1


def norm_0_1(x):
	l, u = x.min(), x.max()
	if u == l:
		raise ValueError("constant image pixel vaues")
	return (x - l) / (u - l)


def generate_crops(dir_infdata,
				   fname_segmasks,
				   min_nuclear_diam=50,
				   width=200,
				   height=200,
				   DO_VIZ=True,
				   viz_dpi=150):
	dir_infdata = Path(dir_infdata)
	fname_segmasks = Path(fname_segmasks)

	# results dirs
	current_filename = Path(os.path.basename(__file__))
	dir_results = Path("inference/results") / os.path.join(
		current_filename.stem, "")
	dir_results.mkdir(exist_ok=True, parents=True)
	dir_viz = Path("inference/viz") / current_filename.stem
	dir_viz.mkdir(exist_ok=True, parents=True)

	if LOG_VERSION_2_RESULTS:
		dir_viz = dir_viz / "version2"
		dir_viz.mkdir(exist_ok=True, parents=True)

	# get masks and the fname of the corresponding FOV nucleus channel
	masks, fnames_nuc = torch.load(fname_segmasks)

	crops_all = []
	df_meta_all = []

	# iterate over the FOVs

	for i, mask in tqdm.tqdm(enumerate(masks), total=len(masks)):
		print("Mask ", i)
		crops = []
		df_meta = pd.DataFrame(columns=[
			"fname_pro", "fname_nuc", "nuc_centroid", "idx_instance",
			"idx_mask", "axis_major_length"
		])

		n_instances = len(np.unique(mask)) - 1
		f_nuc = fnames_nuc[i]
		f_pro = str(f_nuc)[:-7] + "pro.png"
		print(f_pro)
		# if "C3" not in f_pro:
		# 	continue
		try:
			img_nuc = np.array(Image.open(f_nuc))
			img_pro = np.array(Image.open(f_pro))
		except UnidentifiedImageError as e:
			print("caught error")
			print(e)

		if VERSION == 1:
			# normalize the whole FOV
			img_nuc = norm_0_1(img_nuc)
			img_pro = norm_0_1(img_pro)

		if VERSION == 2: 
			# compute an intensity threshold. The `img_pro` is already put in the
			# range [0,255].
			num_pixels_over_thresh = PCNT_PIXELS * height * width

		# within a FOV, iterate over the nucleus instances
		crops_keep, crops_reject = [], []
		for j in range(1, n_instances + 1):
			# compiute the centroid
			target_indices = np.argwhere(mask == j)
			centroid = target_indices.mean(axis=0).astype(int)

			mask_this_idx = np.zeros_like(mask)
			mask_this_idx[mask == j] = 1
			axis_major_length = measure.regionprops(
				mask_this_idx)[0].axis_major_length
			if axis_major_length < min_nuclear_diam:
				continue

			# do image crop, normalize each channel independently.
			# and check that it's not hitting the image border
			slc_0 = slice(centroid[0] - height // 2, centroid[0] + height // 2)
			slc_1 = slice(centroid[1] - width // 2, centroid[1] + width // 2)
			img_pro_crop = img_pro[slc_0, slc_1]
			img_nuc_crop = img_nuc[slc_0, slc_1]
			mask_crop = mask[slc_0, slc_1].copy()
			mask_crop[mask_crop > 0] = 1
			if img_pro_crop.shape != (width, height):
				continue  # skip the crop bc we hit a border
			nuc_dist_map = do_nuclear_dist_transform(mask_crop)

			if VERSION == 2:
				# the skip case 
				if (img_pro_crop>THRESH).sum() < num_pixels_over_thresh:
					crops_reject.append(img_pro_crop.copy())
					continue 
				# the keep case
				else: 
					crops_keep.append(img_pro_crop.copy())

			if VERSION in (0,2):
				img_pro_crop = norm_0_1(img_pro_crop)
				img_nuc_crop = norm_0_1(img_nuc_crop)
			img_crop = np.stack((
				img_pro_crop,
				img_nuc_crop,
				nuc_dist_map,
			))

			# add the crop and metadata
			crops.append(img_crop)
			row_meta = pd.DataFrame(
				dict(
					fname_pro=[f_pro],
					fname_nuc=[f_nuc],
					nuc_centroid=[centroid],
					idx_instance=[j],
					idx_mask=[i],
					axis_major_length=[axis_major_length],
				))
			df_meta = pd.concat([df_meta, row_meta])

		# visualization
		if 1: 
			f_save_original_image = dir_viz / (Path(f_pro).stem+".png")
			Image.fromarray(img_pro, "L").save(f_save_original_image)
			
			nrow=3
			pad_value = 122

			if len(crops_keep) > 0:
				f_save_keep = dir_viz / (Path(f_pro).stem + f"_keep_{len(crops_keep)}.png")
				grid_keep = make_grid(torch.from_numpy(np.stack(crops_keep)).unsqueeze(1), nrow=3, pad_value=pad_value)
				grid_keep = (grid_keep[0]).numpy().astype(np.uint8)
				Image.fromarray(grid_keep, "L").save(f_save_keep)

			if len(crops_reject) != 0:
				f_save_reject = dir_viz / (Path(f_pro).stem + f"_reject_{len(crops_reject)}.png")
				grid_reject = make_grid(torch.from_numpy(np.stack(crops_reject)).unsqueeze(1), nrow=3, pad_value=pad_value)
				grid_reject = (grid_reject[0]).numpy().astype(np.uint8)
				Image.fromarray(grid_reject, "L").save(f_save_reject)

		# optionally visualize crops for this FOV in a grid
		if 0:
			if len(crops) < 1:
				continue

			crops_arr = torch.from_numpy(np.stack(crops))  # (N,2,H,W)

			grid_nuc = make_grid(crops_arr[:, [1]], nrow=8, pad_value=0.5)
			f, axs = plt.subplots()
			axs.imshow(grid_nuc.permute(1, 2, 0))
			f.savefig(dir_viz / (f"{f_nuc.stem}" + "nucleus_only.png"),
					  dpi=viz_dpi)
			plt.close()

			grid_pro = make_grid(crops_arr[:, [0]], nrow=8, pad_value=0.5)
			f, axs = plt.subplots()
			axs.imshow(grid_pro.permute(1, 2, 0))
			f.savefig(dir_viz / (f"{f_nuc.stem}" + "protein_only.png"),
					  dpi=viz_dpi)
			plt.close()

		crops_all.extend(crops)
		df_meta_all.append(df_meta)

	# save the crops
	crops_all_save = torch.from_numpy(
		np.stack(crops_all))  # (n,2,width,height)
	df_meta_all_save = pd.concat(df_meta_all)
	if VERSION > 0:
		f_save_crop = dir_results / f"crops_v{VERSION}.pt"
		f_save_df_meta = dir_results / f"crops_meta_v{VERSION}.csv"
	else:
		f_save_crop = dir_results / f"crops.pt"
		f_save_df_meta = dir_results / f"crops_meta.csv"
	torch.save(crops_all_save, f_save_crop)
	df_meta_all_save.to_csv(f_save_df_meta)


def generate_crops_opencell(min_nuclear_diam=50,
							width=200,
							height=200,
							resize_dim=100,
							DO_VIZ=True,
							viz_dpi=150):
	"""
	The width and height params are the number pixels in the original FOV that 
	we'll crop from.
	The `resize_dim	 will then resize that. 
	"""

	# results dirs
	current_filename = Path(os.path.basename(__file__))
	# dir_results = Path("inference/results") / os.path.join(current_filename.stem, "")
	# dir_results.mkdir(exist_ok=True, parents=True)
	dir_viz = Path("inference/viz") / current_filename.stem / "opencell"
	dir_viz.mkdir(exist_ok=True, parents=True)

	dir_output = Path("data/opencell_crops_processed2")

	dir_data_opencell = Path(
		"/hpc/projects/group.leonetti/opencell-microscopy/")
	dir_masks_opencell = Path("inference/results/nuclear_segmentation/opencell/")
	df_meta_opencell = pd.read_csv(dir_data_opencell / "2021-08-30_good-fovs.csv")

	proteins = np.sort(df_meta_opencell.target_name.unique())
	df_meta_all = []
	# iterate over proteins
	for protein in proteins:
		df_this_prot = df_meta_opencell[df_meta_opencell['target_name'] == protein]
		filepaths = df_this_prot["filepath"]
		crops_all = []

		for idx_mask, filepath in enumerate(filepaths):
			try: 
				fname = dir_data_opencell / filepath
				img_proj = AICSImage(fname).data[0].max(1)
				img_nuc, img_pro = img_proj[0], img_proj[1]

			except:
				continue

			mask = torch.load(dir_masks_opencell / (Path(filepath).stem+".pt"))
			assert mask.shape == img_proj.shape[1:]
			n_instances = len(np.unique(mask))-1

			# in this version, do the normalization at the FOV level
			if VERSION == 1:
				img_nuc = norm_0_1(img_nuc)
				img_pro = norm_0_1(img_pro)

			num_pixels_over_thresh = -1
			if VERSION == 2: 
				# compute an intensity threshold. The `img_pro` is already put in the
				# range [0,255].
				num_pixels_over_thresh = PCNT_PIXELS * height * width

			crops, df_meta = crop_one_fov(mask, img_nuc, img_pro, f_pro=fname, f_nuc=filepath,
				height=height, width=width, idx_mask=idx_mask, num_pixels_over_thresh=num_pixels_over_thresh)
			
			crops_all += crops
			df_meta_all.append(df_meta)
		
		if 1:
			# channels are (protein, nucleus, nuclear transform)
			# load something from Orphans 
			# img_nuc = np.load("data/opencell_crops_proteins_plus_orphans/TSPAN3_nuc.npy")
			# img_pro = np.load("data/opencell_crops_proteins_plus_orphans/TSPAN3_pro.npy")
			# img_nucdist = np.load("data/opencell_crops_proteins_plus_orphans/TSPAN3_nucdist.npy")
			# imgs_inf = np.stack((img_pro, img_nuc, img_nucdist), axis=1)

			import matplotlib.pyplot as plt
			for k in range(5):
				f, axs = plt.subplots(1,3)
				for i in range(3):
					axs[i].imshow(crops[k][i])
					axs[i].set_axis_off()
					# axs[1,i].imshow(imgs_inf[k,i])
				f.savefig(dir_viz / f"crops_sample_{protein}_{k}.png")
				plt.close()
		
		# combine and resize the crops 
		crops_all = np.stack(crops_all)
		crops_all = T.Resize(resize_dim)(torch.from_numpy(crops_all)).float().numpy()

		# generate the labels by just copying it from the old one and adjusting the length
		labels_old = np.load(f"data/opencell_crops_proteins/{protein}_label.npy")
		labels_new = np.array(list([labels_old[0]])*len(crops_all))

		# save 
		np.save(dir_output / f"{protein}_pro", crops_all[:,0])
		np.save(dir_output / f"{protein}_nuc", crops_all[:,1])
		np.save(dir_output / f"{protein}_nucdist", crops_all[:,2])	
		np.save(dir_output / f"{protein}_label", labels_new)

	
	df_meta_all = pd.concat(df_meta_all)
	df_meta_all.to_csv(dir_viz / "df_meta_all.csv")

		

def crop_one_fov(mask, img_nuc, img_pro, f_pro, f_nuc, height, width, 
		num_pixels_over_thresh, idx_mask=-1,):
	assert VERSION in (0,1,2) 
	if VERSION == 1:
		img_nuc = norm_0_1(img_nuc)
		img_pro = norm_0_1(img_pro)

	n_instances = len(np.unique(mask))-1
	crops = []
	df_meta = pd.DataFrame(columns=[
			"fname_pro", "fname_nuc", "nuc_centroid", "idx_instance",
			"idx_mask", "axis_major_length"
		])

	for j in range(1, n_instances + 1):
		# compiute the centroid
		target_indices = np.argwhere(mask == j)
		centroid = target_indices.mean(axis=0).astype(int)

		mask_this_idx = np.zeros_like(mask)
		mask_this_idx[mask == j] = 1
		axis_major_length = measure.regionprops(
			mask_this_idx)[0].axis_major_length
		
		if axis_major_length < min_nuclear_diam:
			continue

		# do image crop, normalize each channel independently.
		# and check that it's not hitting the image border
		slc_0 = slice(centroid[0] - height // 2, centroid[0] + height // 2)
		slc_1 = slice(centroid[1] - width // 2, centroid[1] + width // 2)
		img_pro_crop = img_pro[slc_0, slc_1]
		img_nuc_crop = img_nuc[slc_0, slc_1]
		mask_crop = mask[slc_0, slc_1].copy()
		mask_crop[mask_crop > 0] = 1
		if img_pro_crop.shape != (width, height):
			continue  # skip the crop bc we hit a border
		nuc_dist_map = do_nuclear_dist_transform(mask_crop)

		if VERSION == 2:
			# the skip case  - for opencell, we don't actually do the filtering 
			if (img_pro_crop>THRESH).sum() < num_pixels_over_thresh:
				# crops_reject.append(img_pro_crop.copy())
				continue 
			# the keep case
			else: 
				pass 
				# crops_keep.append(img_pro_crop.copy())

		if VERSION in (0,2):
			img_pro_crop = norm_0_1(img_pro_crop)
			img_nuc_crop = norm_0_1(img_nuc_crop)
		img_crop = np.stack((
			img_pro_crop,
			img_nuc_crop,
			nuc_dist_map,
		))

		# some debugging stuff
		if 0:
			f, axs = plt.subplots(1, 3)
			for i in [0, 1, 2]:
				axs[i].imshow(img_crop[i])
			f.savefig('tmp.png')

		# add the crop and metadata
		crops.append(img_crop)
		row_meta = pd.DataFrame(
			dict(
				fname_pro=[f_pro],
				fname_nuc=[f_nuc],
				nuc_centroid=[centroid],
				idx_instance=[j],
				idx_mask=[idx_mask],
				axis_major_length=[axis_major_length],
			))
		df_meta = pd.concat([df_meta, row_meta])

	return crops, df_meta


def do_nuclear_dist_transform(mask_crop):
	# next two lines are from chatgpt
	boundary = binary_dilation(mask_crop) ^ binary_erosion(mask_crop)
	distance_array = distance_transform_edt(~boundary)

	# normalize distance
	assert distance_array.shape[0] == distance_array.shape[
		1], "if shapes not square, rethink how to normalize"
	distance_array = distance_array / distance_array.shape[0]

	# make the outisde-nucleus stuff negative distance
	distance_array[~(mask_crop.astype(bool))] *= -1

	if 0:
		f, axs = plt.subplots(1, 3)
		axs[0].imshow(mask_crop)
		axs[1].imshow(boundary)
		im = axs[2].imshow(distance_array)
		f.colorbar(im, ax=axs[2])
		f.savefig("tmp.png")

	return distance_array


def save_crops_to_dataset():
	"""
	Save the crops in the style needed to be added to the training dataset. 
	This was for doing training jointly with the OpenCell data.
	"""
	fname_crops = "inference/results/crop/crops.pt"
	fname_crops_meta = "inference/results/crop/crops_meta.csv"
	dir_data = Path("data/opencell_crops_proteins_plus_orphans")

	crops_inf = torch.load(fname_crops)
	crops_inf = T.Resize(100)(crops_inf)

	df_meta_inf = pd.read_csv(fname_crops_meta)
	df_meta_inf['well_id'] = [
		s.split("_")[-3] for s in df_meta_inf['fname_pro']
	]
	fname_annotations = "data/cz_infectedcell_finalwellmapping.csv"
	df_annotations = pd.read_csv(fname_annotations)

	df_meta_inf = df_meta_inf.merge(df_annotations,
									how='left',
									left_on='well_id',
									right_on='well_id_new')
	# the next file we pull is created by get_crop_features.py
	from compare_opencell_targets import get_df_inf_lookup
	df_inf_lookup = get_df_inf_lookup(df_meta_inf)

	proteins = df_inf_lookup.protein.unique()
	proteins = [p for p in proteins if 'pML' not in p]
	for prot in proteins:
		print(prot)
		row_lookup = df_inf_lookup[df_inf_lookup['protein'] == prot]
		well_ids = row_lookup.well_id
		idxs = np.where(df_meta_inf['well_id'].isin(well_ids))[0]
		crops_this = crops_inf[idxs].numpy()

		protein_id = row_lookup['protein_id'].values[0]
		labels = np.array([[protein_id, prot, '']] * len(crops_this))
		np.save(dir_data / f"{prot}_pro.npy", crops_this[:, 0])
		np.save(dir_data / f"{prot}_nuc.npy", crops_this[:, 1])
		np.save(dir_data / f"{prot}_nucdist.npy", crops_this[:, 2])
		np.save(dir_data / f"{prot}_label.npy", labels)

	pass


if __name__ == "__main__":
	# crops for opencell
	if 0:
		min_nuclear_diam = 50  # px
		width, height = 200, 200  # px
		generate_crops_opencell(min_nuclear_diam=min_nuclear_diam,
								width=200,
								height=200,
								DO_VIZ=True,
								viz_dpi=150)

	# crops for orphans (the inference set)
	if 1:
		min_nuclear_diam = 50  # px
		width, height = 200, 200  # px
		dir_infdata = "inference/results/load_inf_data/"
		fname_segmasks = "inference/results/nuclear_segmentation/all_segmasks.pt"
		generate_crops(dir_infdata,
					   fname_segmasks,
					   min_nuclear_diam=min_nuclear_diam,
					   width=width,
					   height=height,
					   DO_VIZ=True)
		# save_crops_to_dataset()



