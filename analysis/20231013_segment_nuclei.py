# options: choose a diameter 

diameter = 120
diameter = None
from pathlib import Path
from PIL import Image
import numpy as np
import ipdb 
import torch
import os

def norm_0_1(x):
	l, u = x.min(), x.max()
	if u == l:
		raise ValueError("constant image pixel vaues")
	return (x-l) / (u-l)

current_filename = Path(os.path.basename(__file__))
results_dir = Path("analysis/results") / current_filename.stem
results_dir.mkdir(exist_ok=True, parents=True)
results_dir_cellpose = results_dir / "cellpose_segs"
results_dir_cellpose.mkdir(exist_ok=True, parents=True)

results_dir_tmp = Path("analysis/tmp")
results_dir_tmp.mkdir(exist_ok=True)

# baseline code from the Colab https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_GPU.ipynb

### our data 
data_maxproj_set48 = Path("analysis/results/20231010_maxproj_set48")
fnames = [l for l in data_maxproj_set48.iterdir()]
fnames_pro = sorted([l for l in fnames if "_pro.png" in str(l)])
fnames_nuc = sorted([l for l in fnames if "_nuc.png" in str(l)])


# imgs_pro = [np.array(Image.open(f)) for f in fnames_pro]
imgs_nuc = [np.array(Image.open(f)) for f in fnames_nuc]

from cellpose import models
model = models.Cellpose(gpu=True, model_type='cyto')
print("Running cellpose")
masks, flows, styles, diams = model.eval(imgs_nuc, diameter=diameter, 
	flow_threshold=None, channels=None)
fname_mask = results_dir / f"all_segmasks.pt"
torch.save([masks, fnames_nuc], fname_mask)
ipdb.set_trace()

import matplotlib.pyplot as plt
# save the mask and also visualize them
for i in range(len(masks)):
	f, axs = plt.subplots(1,2)
	axs[0].imshow(imgs_nuc[i],cmap='gray')
	axs[1].imshow(masks[i], cmap='gray')
	f.savefig(results_dir / f"seg_sample_{fnames[i].stem}.png", dpi=200)
	plt.close()

plt.close()

