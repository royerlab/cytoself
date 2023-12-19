# run in the home directory as python analysis/20231010_validate_embeddings.py
import ipdb
import os 
from pathlib import Path
import numpy as np
import torch
import pandas as pd

# load embeddings from a results dir
RESULTS_DIR = Path("results/20231009_train_all_no_nucdist")
embed_images = torch.from_numpy(np.load(RESULTS_DIR / "embeddings/vqvec2.npy" ))
embed_images = embed_images.view((len(embed_images), -1)) # flatten the array 

# load labels for the test that was saved during datamanager 
DIR_TEST_DATASET_META = Path("data/test_dataset_metadata") 
labels = np.load(DIR_TEST_DATASET_META / "test_dataset_labels.npy", allow_pickle=True)
df = pd.DataFrame(labels, columns=["ensg", "name", "loc_grade"])

# check that embeddings and labels are at least the same dimension
assert len(embed_images) == len(labels)
prots = sorted(df.name.unique())
assert len(prots) == 1311

# consensus embedding as a mean 
embed_consensus = [] 
img_counts = []
localization_labels = []

for prot in prots:
	idxs = np.where(df.name==prot)[0]
	z = embed_images[idxs]
	embed_consensus.append(z.mean(0))
	img_counts.append(len(z))

	localization_label = df[df.name==prot].loc_grade.unique()
	localization_labels.append(localization_label[0])

embed_consensus = torch.stack(embed_consensus) # (n_prot, 1024)
img_counts = torch.tensor(img_counts)
dist = torch.cdist(embed_consensus, embed_consensus)
argsortdist = torch.argsort(dist, dim=1, descending=False)
assert torch.all(argsortdist[:,0] == torch.arange(len(argsortdist)))   # nearest neighbor should be itself
argsortdist = argsortdist[:,1:]

ipdb.set_trace()
pass
# lets do the most basic possible test based on L2 distance which, remember, is not actually the proposed method by Kobayashi et al.
for i in range(20):
	print(localization_labels[i], end= "  ")
	print(localization_labels[argsortdist[i,0].item()], "\t\t\t", localization_labels[argsortdist[i,1].item()])




if __name__=="__main__":
	pass 