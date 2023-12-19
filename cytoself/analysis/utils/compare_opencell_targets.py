"""
Run in project home dir as 
	python inference/compare_opencell_targets.py

For a pretrained model in `<pretrained_model_dir>`, read in the embeddings for 
that model on the OpenCell dataset. This is the dataset the model was trained on

Next get the new inference dataset, its crops, and the embeddings that were 
computed in `inference/get_crop_features.py`. These embedding should have been 
taken from model from `<pretrained_model_dir>`. 
"""

import ipdb
from pathlib import Path 
import torch

def get_nearest_proteins(fname_crops, fname_crops_meta, dir_pretrained_model):
	# results dirs 
	current_filename = Path(os.path.basename(__file__))
	dir_results = Path("inference/results") / current_filename.stem / dir_pretrained_model
	dir_results.mkdir(exist_ok=True, parents=True)
	dir_viz = Path("inference/viz") / current_filename.stem
	dir_viz.mkdir(exist_ok=True, parents=True)

	# get embeddings 
	dir_embeddings = Path("inference/results/get_crop_features") / dir_pretrained_model
	f_embeddings_vqvec2 = dir_embeddings / "embeddings_vqvec2.pt"
	embeddings, labels = torch.load(f_embeddings_vqvec2)
	df_meta = pd.read_csv(dir_embeddings / "crops_meta.csv")

	ipdb.set_trace()
	pass



if __name__=="__main__":
	fname_crops = "inference/results/crop/crops.pt"
	fname_crops_meta = "inference/results/crop/crops_meta.csv"
	dir_pretrained_model = "results/20231011_train_all_no_nucdist"

	get_nearest_proteins(
		fname_crops=fname_crops, 
		fname_crops_meta=fname_crops_meta, 
		dir_pretrained_model=dir_pretrained_model
	)
