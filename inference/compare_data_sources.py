""" 
Want to check that the OpenCell data and the orphan data are close enough:

The FOVs for the inference data are in `inference/results/load_inf_data/`. They 
were constructed in `load_inf_data`. For each channel, they normalize and do max
projection and that's it. 

On the other hand, the FOVs from OpenCell are in /hpc/projects/group.leonetti/opencell-microscopy
""" 
import ipdb 
from aicsimageio import AICSImage
from pathlib import Path
import os 
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torchvision.utils import make_grid

current_filename = Path(os.path.basename(__file__))
results_dir = Path("inference/results") / os.path.join(current_filename.stem, "")
results_dir.mkdir(exist_ok=True, parents=True)

## first let's compare the metadata in the two files 

# a similar protein to A1 is CERS6 
# fname_fov_opencell = Path("/hpc/projects/group.leonetti/opencell-microscopy/public/CERS6_ENSG00000172292/OC-FOV_CERS6_ENSG00000172292_CID001010_FID00013480_stack.tif")
fname_fov_opencell = Path("/hpc/projects/group.leonetti/opencell-microscopy/public/ANKRD46_ENSG00000186106/OC-FOV_ANKRD46_ENSG00000186106_CID001394_FID00033371_stack.tif")
aics_img_opencell = AICSImage(fname_fov_opencell)
data_stack_opencell = aics_img_opencell.data

# get an orphan fov - 3d - well A1, which is ANKRD46, which is ER
# fname_fov_inf = Path("/hpc/instruments/leonetti.dragonfly/infected-cell-microscopy/TICM052-1/raw_data/raw_data_MMStack_9-A1-9.ome.tif")
fname_fov_inf = Path("/hpc/instruments/leonetti.dragonfly/infected-cell-microscopy/TICM052-1/raw_data/raw_data_MMStack_18-A1-18.ome.tif")
aics_img_inf = AICSImage(fname_fov_inf)
data_stack_inf = aics_img_inf.data

print("Pyhsical pixel sizes")
print(f"  OpenCell:  {aics_img_opencell.physical_pixel_sizes}")
print(f"  Inference: {aics_img_inf.physical_pixel_sizes}")
print("Shapes")
print(f"  OpenCell:  {data_stack_opencell.shape}")
print(f"  Inference: {data_stack_inf.shape}")

def norm_0_1(x):
	l, u = x.min(), x.max()
	if u == l:
		raise ValueError("constant image pixel vaues")
	return (x-l) / (u-l)

def get_max_proj(data):
	prot_zstack =  norm_0_1(data[0,1])
	nuc_zstack = norm_0_1(data[0,0])
	prot_proj = prot_zstack.max(0)
	nuc_proj = nuc_zstack.max(0)
	return prot_proj, nuc_proj

prot_proj_opencell, nuc_proj_opencell = get_max_proj(data_stack_opencell)
prot_proj_inf, nuc_proj_inf = get_max_proj(data_stack_inf)

Image.fromarray((prot_proj_opencell*255).astype(np.uint8), "L").save(results_dir / "FOV_prot_ANKRD46_opencell.png")
Image.fromarray((nuc_proj_opencell*255).astype(np.uint8), "L").save(results_dir / "FOV_nuc_ANKRD46_opencell.png")
Image.fromarray((prot_proj_inf*255).astype(np.uint8), "L").save(results_dir / "FOV_prot_ANKRD46.png")
Image.fromarray((nuc_proj_inf*255).astype(np.uint8), "L").save(results_dir / "FOV_nuc_ANKRD46.png")


## crop comparisons
results_idx = 7
results_dir = results_dir / f"compare_{results_idx}"
results_dir.mkdir(exist_ok=True)

# variable `VERSION` is 0 if norming each crop, and 1 if norming each FOV
if results_idx == 0:
	target_prot_opencell = "ANKRD46"
	target_well_inf = "A1"
	VERSION = 0
elif results_idx == 1:
	# the lysosome ones 
	target_prot_opencell = "SLC37A3"
	target_well_inf = "F3" # TSPAN3
	VERSION = 0
elif results_idx == 2:
	# same as results_idx 0 except with version 1 for norming the whol fov
	target_prot_opencell = "ANKRD46"
	target_well_inf = "A1"
	VERSION = 1
elif results_idx == 3:
	# some nucleolus proteins with the old normalization
	target_prot_opencell = "AATF"
	target_well_inf = "C3"
	VERSION = 0
elif results_idx == 4:
	# 
	target_prot_opencell = "TMEM192"
	target_well_inf = "F11" # TSPAN6
	VERSION = 0
elif results_idx == 5:
	target_prot_opencell = "TMEM192" 
	target_well_inf = "F11" # TSPAN6
	VERSION = 1
elif results_idx == 6:
	# 
	target_prot_opencell = "TMEM192"
	target_well_inf = "F3" # TSPAN6
	VERSION = 0
elif results_idx == 7:
	target_prot_opencell = "TMEM192" # TSPAN6
	target_well_inf = "F3"
	VERSION = 1
	

# get the right inference crops based on the version. 
if VERSION == 0:
	crops = torch.load("inference/results/crop/crops.pt")
	crops_meta = pd.read_csv("inference/results/crop/crops_meta.csv")
elif VERSION == 1:
	crops = torch.load("inference/results/crop/crops_v1.pt")
	crops_meta = pd.read_csv("inference/results/crop/crops_meta.csv")
else: 
	raise ValueError()

crops_meta['well_id'] = [row.split("/")[-1].split("_")[2] for row in crops_meta['fname_pro']]
idxs_target_inf = np.where(crops_meta['well_id']==target_well_inf)[0]
crops_target_inf = crops[idxs_target_inf]
f,axs = plt.subplots(1,4)
_ = [axs[i].imshow(crops_target_inf[0,i], cmap='gray') for i in range(3)]
axs[3].imshow(np.zeros_like(crops_target_inf[0,0]))
f.savefig(results_dir / "sample_crop_inf_img.png")

nrow, ncol = 8,8
nimgs = nrow*ncol
# make a grid for inf 
grid = make_grid(crops_target_inf[:nimgs], nrow=nrow).permute(1,2,0).numpy()
Image.fromarray((grid[...,0]*255).astype(np.uint8), "L").save(results_dir / "crops_prot_inf.png")
Image.fromarray((grid[...,1]*255).astype(np.uint8), "L").save(results_dir / "crops_nuc_inf.png")
Image.fromarray((norm_0_1(grid[...,2])*255).astype(np.uint8), "L").save(results_dir / "crops_nucdist_inf.png")

## for opencell first figure out which crops file has the target protein we want 
labels_all = []
for i in range(10):
	labels = pd.read_csv(f"data/opencell_crops/label_data{i:02d}.csv", low_memory=False)
	labels['file'] = i
	labels_all.append(labels)
labels_all = pd.concat(labels_all)
idx_crop_file = labels_all[labels_all['name']==target_prot_opencell]['file'].unique()[0]
# it's 6, so get that crop - getting this npy file is slow
crops_opencell_this = np.load(f"data/opencell_crops/image_data{idx_crop_file:02d}.npy")
labels_opencell_this = pd.read_csv(f"data/opencell_crops/label_data{idx_crop_file:02d}.csv", low_memory=False)

# opencell
idxs_target_opencell = np.where(labels_opencell_this['name']==target_prot_opencell)[0]
crops_target_opencell = crops_opencell_this[idxs_target_opencell]
# inf 
f,axs = plt.subplots(1,4)
_ = [axs[i].imshow(crops_target_opencell[0,...,i], cmap='gray') for i in range(4)]
f.savefig(results_dir / "sample_crop_opencell_img.png")

grid = make_grid(torch.from_numpy(crops_target_opencell[:nimgs]).permute(0,3,1,2), nrow=nrow).permute(1,2,0).numpy()
Image.fromarray((grid[...,0]*255).astype(np.uint8), "L").save(results_dir / "crops_prot_opencell.png")
Image.fromarray((grid[...,1]*255).astype(np.uint8), "L").save(results_dir / "crops_nuc_opencell.png")
Image.fromarray((norm_0_1(grid[...,2])*255).astype(np.uint8), "L").save(results_dir / "crops_nucdist_opencell.png")


for i in range(len(crops_target_opencell)):
	crops_target_opencell[i,...,0] = norm_0_1(crops_target_opencell[i,...,0])
	crops_target_opencell[i,...,1] = norm_0_1(crops_target_opencell[i,...,1])
grid = make_grid(torch.from_numpy(crops_target_opencell[:nimgs]).permute(0,3,1,2), nrow=nrow).permute(1,2,0).numpy()
Image.fromarray((grid[...,0]*255).astype(np.uint8), "L").save(results_dir / "crops_prot_opencell_normed.png")
Image.fromarray((grid[...,1]*255).astype(np.uint8), "L").save(results_dir / "crops_nuc_opencell_normed.png")

ipdb.set_trace()
pass 










