"""
I re-processed the OpenCell crops to be data "data/opencell_crops_processed2"
"""
import ipdb
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch

current_filename = Path(os.path.basename(__file__))
results_dir = Path("analysis/results") / current_filename.stem
results_dir.mkdir(exist_ok=True, parents=True)

nrow, ncol = 8, 8
nimgs = nrow * ncol

EXP = 0
for EXP in [0,1,2]:
	if EXP == 0:
		# both are actually `ANKRD46`
		prot_opencell = "ANKRD46"
		target_well_inf = "A1" # ANKRD46
	elif EXP == 1:
		# two random ER proteins
		prot_opencell = "ATL3"
		target_well_inf = "E2" # C4orf3
	elif EXP == 2:
		# two lysosomal proteins
		prot_opencell = "ARFGAP1"
		target_well_inf = "F11" # "TSPAN6"
	else:	
		raise 

	# opencell crops for ANKRD46
	dir_opencell_crops = Path("data/opencell_crops_processed2")

	img_pro_opencell = np.load(dir_opencell_crops / f"{prot_opencell}_pro.npy")
	img_nuc_opencell = np.load(dir_opencell_crops / f"{prot_opencell}_nuc.npy")
	img_nucdist_opencell = np.load(dir_opencell_crops /
	                               f"{prot_opencell}_nucdist.npy")
	img_nucdist_opencell = np.load(dir_opencell_crops /
	                               f"{prot_opencell}_nucdist.npy")
	img_nucdist_opencell = (img_nucdist_opencell+1)/2 # for viz only
	img_opencell = np.stack(
	    (img_pro_opencell, img_nuc_opencell, img_nucdist_opencell), axis=1)

	#  opencell
	idxs = np.arange(len(img_opencell))
	idxs = idxs[:nimgs]
	np.random.shuffle(idxs)
	grid_opencell = make_grid(
	    torch.from_numpy(img_opencell)[idxs], ncol)

	grid_opencell = (grid_opencell*255).numpy().astype(np.uint8).transpose(1,2,0)
	Image.fromarray(grid_opencell[...,0], "L").save(results_dir / f"exp{EXP}_grid_pro_opencell.png")
	Image.fromarray(grid_opencell[...,1], "L").save(results_dir / f"exp{EXP}_grid_nuc_opencell.png")
	Image.fromarray((grid_opencell[...,2]/255)+127.5, "L").save(results_dir / f"exp{EXP}_grid_nucdist_opencell.png")


	#### orphans
	VERSION = 0
	# version 0 has crop-level norm, version 1 has protein level norm
	if VERSION == 0:
	    crops = torch.load("inference/results/crop/crops.pt")
	    crops_meta = pd.read_csv("inference/results/crop/crops_meta.csv")
	elif VERSION == 1:
	    crops = torch.load("inference/results/crop/crops_v1.pt")
	    crops_meta = pd.read_csv("inference/results/crop/crops_meta.csv")
	else:
	    raise ValueError()

	crops_meta['well_id'] = [
	    row.split("/")[-1].split("_")[2] for row in crops_meta['fname_pro']
	]
	idxs_target_inf = np.where(crops_meta['well_id'] == target_well_inf)[0]
	crops_target_inf = crops[idxs_target_inf]
	crops_target_inf[:,2] = (crops_target_inf[:,2]+1)/2

	idxs = np.arange(len(crops_target_inf))
	np.random.shuffle(idxs)
	idxs = idxs[:nimgs]
	grid_inf = make_grid(crops_target_inf[idxs], nrow).permute(1, 2, 0)
	grid_inf = (grid_inf*255).numpy().astype(np.uint8)
	Image.fromarray(grid_inf[...,0], "L").save(results_dir / f"exp{EXP}_grid_pro_inf.png")
	Image.fromarray(grid_inf[...,1], "L").save(results_dir / f"exp{EXP}_grid_nuc_inf.png")
	Image.fromarray((grid_inf[...,2]/255)+127.5, "L").save(results_dir / f"exp{EXP}_grid_nucdist_inf.png")

ipdb.set_trace()
pass 

