import ipdb
import os 
from pathlib import Path 
import sys 
import numpy as np
import pandas as pd 
from aicsimageio import AICSImage
from PIL import Image

current_filename = Path(os.path.basename(__file__))
results_dir = Path("analysis/results") / current_filename.stem
results_dir.mkdir(exist_ok=True, parents=True)

def norm_0_1(x):
	l, u = x.min(), x.max()
	if u == l:
		raise ValueError("constant image pixel vaues")
	return (x-l) / (u-l)

## what are the `pixel sizes` ?
# Opencell `/clean` stuff
dir_data = Path("/hpc/projects/group.leonetti/opencell-microscopy/")
df_meta = pd.read_csv(dir_data / "2021-08-30_good-fovs.csv") 
dir_seg = Path("inference/results/nuclear_segmentation/opencell/")
fname = dir_data / df_meta.iloc[0]['filepath']
img_ = AICSImage(fname)
img0_proj_nuc = img_.data[0,0].max(0)
img0_cropped_proj_nuc = img0_proj_nuc[:600,:600]
print("Shape       ", img_.shape)
print("Pixel sizes ", img_.physical_pixel_sizes)
print("Meta        ", img_.metadata)

# OpenCell in the `public/` repo
fname = "/hpc/projects/group.leonetti/opencell-microscopy/public/AAMP_ENSG00000127837/OC-FOV_AAMP_ENSG00000127837_CID001050_FID00013888_stack.tif"
img_ = AICSImage(fname)
img1_proj_nuc = img_.data[0,0].max(0)
print("Pixel sizes ", img_.physical_pixel_sizes)
print("Meta        ", img_.metadata)
print("Shape       ", img_.shape) # (1, 2, 51, 600, 600) 
print("Pixel sizes ", img_.physical_pixel_sizes) # PhysicalPixelSizes(Z=0.5, Y=0.2, X=0.2)
print("Meta        ", img_.metadata)

# save 
Image.fromarray((norm_0_1(img0_proj_nuc)*255).astype(np.uint8), "L").save(results_dir / "img0_proj_nuc.png")
Image.fromarray((norm_0_1(img0_cropped_proj_nuc)*255).astype(np.uint8), "L").save(results_dir / "img0_cropped_proj_nuc.png")
Image.fromarray((norm_0_1(img1_proj_nuc)*255).astype(np.uint8), "L").save(results_dir / "img1_proj_nuc.png")

###


ipdb.set_trace()
pass 
