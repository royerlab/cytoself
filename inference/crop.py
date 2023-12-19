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

def norm_0_1(x):
	l, u = x.min(), x.max()
	if u == l:
		raise ValueError("constant image pixel vaues")
	return (x-l) / (u-l)


def generate_crops(dir_infdata, fname_segmasks, min_nuclear_diam=50, 
		width=200, height=200, DO_VIZ=True, viz_dpi=150):
	dir_infdata = Path(dir_infdata)
	fname_segmasks = Path(fname_segmasks)

	# results dirs 
	current_filename = Path(os.path.basename(__file__))
	dir_results = Path("inference/results") / os.path.join(current_filename.stem, "")
	dir_results.mkdir(exist_ok=True, parents=True)
	dir_viz = Path("inference/viz") / current_filename.stem
	dir_viz.mkdir(exist_ok=True, parents=True)

	# get masks and the fname of the corresponding FOV nucleus channel
	masks, fnames_nuc = torch.load(fname_segmasks)

	crops_all = []
	df_meta_all = []

	# iterate over the FOVs 

	for i, mask in enumerate(masks):
		print("Mask ", i)
		crops = []
		df_meta = pd.DataFrame(columns=["fname_pro", "fname_nuc", 
			"nuc_centroid", "idx_instance", "idx_mask", "axis_major_length"])

		n_instances = len(np.unique(mask))-1
		f_nuc = fnames_nuc[i]
		f_pro = str(f_nuc)[:-7] + "pro.png"
		try:
			img_nuc = np.array(Image.open(f_nuc))
			img_pro = np.array(Image.open(f_pro))
		except UnidentifiedImageError as e:
			print("caught error")
			print(e)

		# within a FOV, iterate over the nucleus instances
		for j in range(1,n_instances+1): 
			# compiute the centroid
			target_indices = np.argwhere(mask == j)
			centroid = target_indices.mean(axis=0).astype(int)

			mask_this_idx = np.zeros_like(mask)
			mask_this_idx[mask==j] = 1
			axis_major_length = measure.regionprops(mask_this_idx)[0].axis_major_length
			if axis_major_length < min_nuclear_diam:
				continue

			# do image crop, normalize each channel independently. 
			# and check that it's not hitting the image border
			slc_0 = slice(centroid[0] - height//2, centroid[0] + height//2)
			slc_1 = slice(centroid[1] - width//2, centroid[1] + width//2)
			img_pro_crop = img_pro[slc_0, slc_1]
			img_nuc_crop = img_nuc[slc_0, slc_1]
			mask_crop = mask[slc_0, slc_1].copy()
			mask_crop[mask_crop>0] = 1
			if img_pro_crop.shape != (width, height):
				continue  # skip the crop bc we hit a border
			nuc_dist_map = do_nuclear_dist_transform(mask_crop)

			img_crop = np.stack((
				norm_0_1(img_pro_crop),
				norm_0_1(img_nuc_crop),
				nuc_dist_map,
				))			

			# some debugging stuff
			if 0:
				f,axs = plt.subplots(1,3)
				for i in [0,1,2]:
					axs[i].imshow(img_crop[i])	
				f.savefig('tmp.png')
			
			# add the crop and metadata 
			crops.append(img_crop) 
			row_meta = pd.DataFrame(dict(
				fname_pro=[f_pro],
				fname_nuc=[f_nuc],
				nuc_centroid=[centroid],
				idx_instance=[j],
				idx_mask=[i],
				axis_major_length=[axis_major_length],
				))
			df_meta = pd.concat([df_meta, row_meta])


		# optionally visualize crops for this FOV in a grid
		if 1:
			if len(crops) < 1:
				continue
			
			crops_arr = torch.from_numpy(np.stack(crops)) # (N,2,H,W)

			grid_nuc = make_grid(crops_arr[:,[1]], nrow=8, pad_value=0.5)
			f, axs = plt.subplots()
			axs.imshow(grid_nuc.permute(1,2,0))
			f.savefig(dir_viz/ (f"{f_nuc.stem}" + "nucleus_only.png"), dpi=viz_dpi)
			plt.close()

			grid_pro = make_grid(crops_arr[:,[0]], nrow=8, pad_value=0.5)
			f, axs = plt.subplots()
			axs.imshow(grid_pro.permute(1,2,0))
			f.savefig(dir_viz / (f"{f_nuc.stem}" + "protein_only.png"), dpi=viz_dpi)
			plt.close()

		crops_all.extend(crops)
		df_meta_all.append(df_meta)

	# save the crops
	crops_all_save = torch.from_numpy(np.stack(crops_all)) # (n,2,width,height)
	df_meta_all_save = pd.concat(df_meta_all)
	f_save_crop = dir_results / "crops.pt"
	torch.save(crops_all_save, f_save_crop)
	f_save_df_meta = dir_results / "crops_meta.csv"
	df_meta_all_save.to_csv(f_save_df_meta)


def do_nuclear_dist_transform(mask_crop):
	# next two lines are from chatgpt
	boundary = binary_dilation(mask_crop) ^ binary_erosion(mask_crop)
	distance_array = distance_transform_edt(~boundary)

	# normalize distance 
	assert distance_array.shape[0]==distance_array.shape[1], "if shapes not square, rethink how to normalize"
	distance_array = distance_array / distance_array.shape[0]

	# make the outisde-nucleus stuff negative distance
	distance_array[~(mask_crop.astype(bool))] *= -1

	if 0:
		f, axs = plt.subplots(1,3)
		axs[0].imshow(mask_crop)
		axs[1].imshow(boundary)
		im=axs[2].imshow(distance_array)
		f.colorbar(im, ax=axs[2])
		f.savefig("tmp.png")
	
	return distance_array  

if __name__=="__main__":
	dir_infdata = "inference/results/load_inf_data/"
	fname_segmasks = "inference/results/nuclear_segmentation/all_segmasks.pt"
	min_nuclear_diam = 50 # px
	width, height = 200, 200 # px
	generate_crops(dir_infdata, fname_segmasks, min_nuclear_diam=min_nuclear_diam, 
		width=width, height=height, DO_VIZ=True)

