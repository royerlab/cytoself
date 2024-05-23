"""
Run in project home dir as 
	python inference/nuclear_segmentation.py
Or load as a module and run segment_nuclei_fovs()

Get the data processed FOV files that were prepared by `load_inf_data.py`, which 
should be sitting in `inference/results/load_inf_data/`. They are 2d max projections

Do instance segmentation of the nucleus with CellPose and save the results to a 
single file: `inference/results/nucelear_segmentation/all_segmasks.pt`.

The segmentation maps can be inspected in `inference/results/load_inf_data/`. 
"""

from pathlib import Path
from PIL import Image
import numpy as np
import ipdb 
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from aicsimageio import AICSImage
import tqdm

def segment_nuclei_fovs(dir_data, diameter=None, batch_size=2048, is_opencell=False):
	"""
	diameter is the parameter passed to Cellpose segmentation. By deafult we keep
	it None.
	"""

	dir_data = Path(dir_data)
	fnames = [l for l in dir_data.iterdir()]
	# results dirs 
	current_filename = Path(os.path.basename(__file__))
	dir_results = Path("inference/results") / os.path.join(current_filename.stem, "")
	if is_opencell: 
		dir_results = dir_results / "opencell"
	dir_results.mkdir(exist_ok=True, parents=True)
	dir_viz = Path("inference/viz") / current_filename.stem
	if is_opencell:
		dir_viz = dir_viz / "opencell"
	dir_viz.mkdir(exist_ok=True, parents=True)

	# baseline CellPose code from: https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_GPU.ipynb
	
	# fnames_pro = sorted([l for l in fnames if "_pro.png" in str(l)])
	fnames_nuc = sorted([l for l in fnames if "_nuc.png" in str(l)])

	# imgs_pro = [np.array(Image.open(f)) for f in fnames_pro]
	imgs_nuc = [np.array(Image.open(f)) for f in fnames_nuc]


	from cellpose import models
	is_cuda = torch.cuda.is_available()
	if not is_cuda:
		raise "No GPU found. If that is intentional, then comment out this line" \
			"and change the next line to have `models.Cellpose(gpu=False, model_type='cyto')`"
	model = models.Cellpose(gpu=True, model_type='cyto')
	print("Running cellpose ... may take a minute ")

	masks, flows, styles, diams = model.eval(imgs_nuc, diameter=diameter, 
		flow_threshold=None, channels=None, batch_size=batch_size)

	# save the masks 
	fname_mask = dir_results / f"all_segmasks.pt"
	torch.save([masks, fnames_nuc], fname_mask)

	# visualize the masks
	import matplotlib.pyplot as plt
	for i in range(len(masks)):
		f, axs = plt.subplots(1,2)
		axs[0].imshow(imgs_nuc[i],cmap='gray')
		axs[1].imshow(masks[i], cmap='gray')
		f.savefig(dir_viz / f"seg_sample_{fnames[i].stem}.png", dpi=200)
		plt.close()

def segment_opencell():
	is_opencell=True
	dir_data = Path("/hpc/projects/group.leonetti/opencell-microscopy/")
	df_meta = pd.read_csv(dir_data / "2021-08-30_good-fovs.csv")
	fnames = [dir_data / f for f in df_meta.filepath.values]

	# results dirs 
	current_filename = Path(os.path.basename(__file__))
	dir_results = Path("inference/results") / os.path.join(current_filename.stem, "")
	if is_opencell: 
		dir_results = dir_results / "opencell"
	dir_results.mkdir(exist_ok=True, parents=True)
	dir_viz = Path("inference/viz") / current_filename.stem
	if is_opencell:
		dir_viz = dir_viz / "opencell"
	dir_viz.mkdir(exist_ok=True, parents=True)

	from cellpose import models
	is_cuda = torch.cuda.is_available()
	if not is_cuda:
		raise "No GPU found. If that is intentional, then comment out this line" \
			"and change the next line to have `models.Cellpose(gpu=False, model_type='cyto')`"
	model = models.Cellpose(gpu=True, model_type='cyto')

	for file in tqdm.tqdm(fnames): 
		try: 
			img = AICSImage(file).data
		except:
			print("Didn't find file ", file, " skipping ")
			continue
		img_nuc_proj = img[0,0].max(0)
		img_nuc_proj = (img_nuc_proj / img_nuc_proj.max() * 255).astype(np.uint8)	
		img_nuc = img_nuc_proj.copy()

		diameter=None
		batch_size=4
		masks, flows, styles, diams = model.eval(img_nuc, diameter=diameter, 
			flow_threshold=None, channels=None, batch_size=batch_size)

		f_save = dir_results / f"{file.stem}.pt"
		torch.save(masks, f_save)

		# f,axs = plt.subplots(1,2)	
		# axs[0].imshow(img_nuc_proj)
		# axs[1].imshow(masks)
		# f.savefig("tmp.png")

if __name__ == "__main__":
	# orphan proteins 
	if 0:
		dir_data = Path("inference/results/load_inf_data/")
		is_opencell = False
		segment_nuclei_fovs(dir_data, is_opencell)

	segment_opencell()

