"""
Run in project home dir as 
	python inference/load_inf_data.py
Or load as a module and run load_inf_dataset52()

Function: get raw data FOVs into a standard format and save to 
`inference/results/load_inf_data`. Each channel is saved as a separate grayscale
png image.

The standard format is max-intensity projection, channel 0 protein, channel 
1 nucleus, and normalize to [0,1]. If things like denoising or max-intensity 
supression were needed, this would be the spot for that preprocessing.

Also save a DataFrame with  metadata about each FOV. 

Also, visualize separate channels in `inference/viz/load_inf_data`

The following file had a problem: 
	/hpc/instruments/leonetti.dragonfly/infected-cell-microscopy/TICM052-1/raw_data/raw_data_MMStack_16-A1-16.ome.tif
"""
import ipdb 
from PIL import Image
from aicsimageio import AICSImage
from pathlib import Path
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from skimage.io import imread, imsave
from skimage import color
from PIL import Image


def norm_0_1(x):
	l, u = x.min(), x.max()
	if u == l:
		raise ValueError("constant image pixel vaues")
	return (x-l) / (u-l)

def load_inf_dataset52(dir_fovs, DO_VIZ=True):
	dir_fovs = Path(dir_fovs)
	
	# make directories for storing results
	current_filename = Path(os.path.basename(__file__))
	dir_results = Path("inference/results") / os.path.join(current_filename.stem, "")
	dir_results.mkdir(exist_ok=True, parents=True)
	dir_viz = Path("inference/viz") / current_filename.stem
	dir_viz.mkdir(exist_ok=True, parents=True)

	fnames = [f for f in dir_fovs.glob("*.tif")] 
	print(f"Num files {len(fnames)}")

	# construct a dataframe for the image metadata
	df_imgs_meta = pd.DataFrame(columns=["fname", "well_id", "fov_id", "is_green"])

	# create the metadata dataframe before reading the images
	for fname in fnames:
		# metadata structure  "raw_data_MMStack_592-B8-16.ome.tif". Well=B8, FOV=16
		well_id, fov_id = fname.stem.split(".")[0].split("-")[-2:]

		row = pd.DataFrame(dict(
				fname=[fname],
				well_id=[well_id],
				fov_id=[fov_id],
				is_green=[1],
		))
		df_imgs_meta = pd.concat([df_imgs_meta, row], ignore_index=True)

	# now read in the iamges, create the FOVs for the next stage, and visualize
	for idx, row in df_imgs_meta.iterrows():
		print(idx, fname)
		fname = row.fname 

		# there was an issue with this one file 
		if fname.stem == "raw_data_MMStack_1-A1-1.ome":
			continue 
		aics_img = AICSImage(fname)
		x = aics_img.data # (1, 2, Z, H, W)
		assert x.ndim == 5 and x.shape[:2] == (1,2)

		# max intensity projection for the nucleus and protein channel independently
		x_pro_map = x[0,1].max(0)
		x_nuc_map = x[0,0].max(0)

		x_pro_map = norm_0_1(x_pro_map)
		x_nuc_map = norm_0_1(x_nuc_map)
		
		# save as 1-channel png
		fname_nuc = dir_results / f"max_proj_{row.well_id}_{row.fov_id}_nuc.png"
		fname_pro = dir_results / f"max_proj_{row.well_id}_{row.fov_id}_pro.png"
		Image.fromarray((x_nuc_map*255).astype(np.uint8), mode="L").save(fname_nuc)
		Image.fromarray((x_pro_map*255).astype(np.uint8), mode="L").save(fname_pro)

		# viewing 
		if DO_VIZ:
			f, axs = plt.subplots(1,2)
			axs[0].imshow(x_nuc_map, cmap='gray')
			axs[1].imshow(x_pro_map, cmap='gray')
			axs[0].set(title="nucleus")
			axs[1].set(title="protein")
			fname_save = dir_viz / (fname.stem + "_max_proj.png")
			f.savefig(fname_save, dpi=200)
			plt.close()

		del aics_img

	ipdb.set_trace()
	pass 

if __name__ == "__main__":
	dir_fovs = "/hpc/instruments/leonetti.dragonfly/infected-cell-microscopy/TICM052-1/raw_data"
	load_inf_dataset52(dir_fovs=dir_fovs, DO_VIZ=True)
