# should be run in the project dir for cytoself
# for developing the inference pipeline, let's process the images 

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

# make the results_dir 
current_filename = Path(os.path.basename(__file__))
results_dir = Path("analysis/results") / current_filename.stem
results_dir.mkdir(exist_ok=True, parents=True)

results_maxproj_set48 = Path("analysis/results/20231010_maxproj_set48")
results_maxproj_set48.mkdir(exist_ok=True, parents=True)

DATA_DIR = Path("/hpc/instruments/leonetti.dragonfly/infected-cell-microscopy/TICM048-1/raw_data")
fnames = [f for f in DATA_DIR.glob("*.tif")] 
print(f"Num files {len(fnames)}")

# get the orphan proteins in `df_exp` and filter for the ones with good signal have col is_greed
fname_exp = Path("data/orphan_protein.csv")
df_exp = pd.read_csv(fname_exp)
df_exp_green = df_exp[df_exp['is_green']==1]

# construct a dataframe for the image metadata
df_imgs_meta = pd.DataFrame(columns=["fname", "well_id", "fov_id", "is_green"])
for fname in fnames:

	# metadata structure  "raw_data_MMStack_592-B8-16.ome.tif". Well=B8, FOV=16
	well_id, fov_id = fname.stem.split(".")[0].split("-")[-2:]

	idxs_well_id = np.where(well_id == df_exp['well_id'].values)[0]
	if len(idxs_well_id)==0:
		print(f"did not find meta for well_id={well_id}")
	elif len(idxs_well_id)>1:
		print(f"More than one match for well_id={well_id}")
	else:
		assert len(idxs_well_id) == 1
		row = pd.DataFrame(dict(
				fname=[fname],
				well_id=[well_id],
				fov_id=[fov_id],
				is_green=[df_exp.iloc[idxs_well_id[0]]['is_green']],
		))
		df_imgs_meta = pd.concat([df_imgs_meta, row], ignore_index=True)

df_imgs_meta_green = df_imgs_meta[df_imgs_meta['is_green']==1]
print(f"Number of zstacks that are 'green' {len(df_imgs_meta_green)}")
print(f"Number of unique wells that are 'green' {len(df_imgs_meta_green.well_id.unique())}")

# now get the zstacks for the green images
for idx, row in df_imgs_meta_green.iterrows():
	fname = row.fname 
	aics_img = AICSImage(fname)
	# ipdb.set_trace()
	x = aics_img.data # (1, 2, Z, H, W)
	assert x.ndim == 5 and x.shape[:2] == (1,2)

	# max intensity projection for the nucleus and protein channel independently
	x_pro_map = x[0,1].max(0)
	x_nuc_map = x[0,0].max(0)

	x_pro_map = norm_0_1(x_pro_map)
	x_nuc_map = norm_0_1(x_nuc_map)
	
	# save as 1-channel png
	fname_nuc = results_maxproj_set48 / f"max_proj_{row.well_id}_{row.fov_id}_nuc.png"
	fname_pro = results_maxproj_set48 / f"max_proj_{row.well_id}_{row.fov_id}_pro.png"
	Image.fromarray((x_nuc_map*255).astype(np.uint8), mode="L").save(fname_nuc)
	Image.fromarray((x_pro_map*255).astype(np.uint8), mode="L").save(fname_pro)

	# viewing 
	f, axs = plt.subplots(1,2)
	axs[0].imshow(x_nuc_map, cmap='gray')
	axs[1].imshow(x_pro_map, cmap='gray')
	axs[0].set(title="nucleus")
	axs[1].set(title="protein")
	fname_save = results_dir / (fname.stem + "_max_projs.png")
	f.savefig(fname_save, dpi=200)
	plt.close()

ipdb.set_trace()
pass 