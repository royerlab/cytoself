from pathlib import Path
import ipdb 
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from PIL import Image
import os

current_filename = Path(os.path.basename(__file__))
results_dir = Path("analysis/results") / current_filename.stem

fname_mask = Path('analysis/results/20231013_segment_nuclei/all_segmasks.pt')
masks, fnames_nuc = torch.load(fname_mask)

width, height = 200, 200
crops_all = []
df_meta_all = []

def norm_0_1(x):
	l, u = x.min(), x.max()
	if u == l:
		raise ValueError("constant image pixel vaues")
	return (x-l) / (u-l)

for i, mask in enumerate(masks):
	crops = []
	df_meta = pd.DataFrame(columns=["fname_pro", "fname_nuc", "cell_centroid", "mask_idx"])

	n_instances = len(np.unique(mask))-1
	f_nuc = fnames_nuc[i]
	f_pro = str(f_nuc)[:-7] + "pro.png"
	img_nuc = np.array(Image.open(f_nuc))
	img_pro = np.array(Image.open(f_pro))

	for j in range(1,n_instances+2): 
		target_indices = np.argwhere(mask == j)
		centroid = target_indices.mean(axis=0).astype(int)

		# do image crop, normalize each channel independently. 
		# and check that it's not hitting the image border
		# ipdb.set_trace()
		slc_0 = slice(centroid[0] - height//2, centroid[0] + height//2)
		slc_1 = slice(centroid[1] - width//2, centroid[1] + width//2)
		img_pro_crop = img_pro[slc_0, slc_1]
		img_nuc_crop = img_nuc[slc_0, slc_1]
		if img_pro_crop.shape != (width, height):
			continue  # skip the crop bc we hit a border
		img_crop = np.stack((
			norm_0_1(img_pro_crop),
			norm_0_1(img_nuc_crop),
			))			
		
		# add the crop and metadata 
		crops.append(img_crop) 
		row_meta = pd.DataFrame(dict(
			fname_pro=[f_pro],
			fname_nuc=[f_nuc],
			cell_centroid=[centroid],
			mask_idx=[j],
			))
		df_meta = pd.concat([df_meta, row_meta])
		

	# optionally visualize crops in a grid
	if 1:
		if len(crops) < 1:
			continue
		from torchvision.utils import make_grid
		crops_arr = torch.from_numpy(np.stack(crops)) # (N,2,H,W)

		grid_nuc = make_grid(crops_arr[:,[1]], nrow=8, pad_value=0.5)
		f, axs = plt.subplots()
		axs.imshow(grid_nuc.permute(1,2,0))
		f.savefig(results_dir / (f"{f_nuc.stem}" + "nucleus_only.png"))
		plt.close()

		grid_pro = make_grid(crops_arr[:,[0]], nrow=8, pad_value=0.5)
		f, axs = plt.subplots()
		axs.imshow(grid_pro.permute(1,2,0))
		f.savefig(results_dir / (f"{f_nuc.stem}" + "protein_only.png"))
		plt.close()


		# # for visualization, put  protein in green, the nucleus in red for RGBs
		# imgs_rgb_tmp = torch.from_numpy(np.stack(crops)).clone()
		# imgs_rgb = torch.zeros((len(imgs_rgb_tmp), 3, *imgs_rgb_tmp.shape[2:]))
		# imgs_rgb[:,1] = imgs_rgb_tmp[:,0]
		# imgs_rgb[:,2] = imgs_rgb_tmp[:,1]

		# grid = make_grid(imgs_rgb, nrow=5)
		# f, axs = plt.subplots()
		# axs.imshow(grid.permute(1,2,0))
		# fname_save = results_dir / f""
		# f.savefig()
		# # gri
	crops_all.extend(crops)
	df_meta_all.append(df_meta)



ipdb.set_trace()
pass
