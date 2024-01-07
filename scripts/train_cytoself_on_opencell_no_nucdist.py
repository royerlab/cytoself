from os.path import join
import ipdb

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import tqdm
import json
from pathlib import Path

from cytoself.analysis.analysis_opencell import AnalysisOpenCell
from cytoself.datamanager.opencell import DataManagerOpenCell
from cytoself.datamanager.preloaded_dataset import PreloadedDataset
from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer
from cytoself.trainer.utils.plot_history import plot_history_cytoself

# declare results dir 
results_dir = Path("results/20231011_train_all_no_nucdist")
results_dir = Path("results/20231025_train_all_no_nucdist")
results_dir = Path("results/20231218_train_all_no_nucdist")
results_dir = Path("results/tmp")
results_dir.mkdir(exist_ok=True) 
tensorboard_path = "logs"

# 1. Prepare Data
# data_ch = ['pro', 'nuc', 'nucdist'] # ['pro', 'nuc', 'nucdist']
data_ch = ['pro', 'nuc'] # 

datapath = Path("data/opencell_crops_proteins/")

# DataManagerOpenCell.download_sample_data(datapath)  # donwload data
datamanager = DataManagerOpenCell(datapath, data_ch, fov_col=None)
datamanager.const_dataloader(batch_size=32, label_name_position=1)


# 2. Create and train a cytoself model
model_args = {
    'input_shape': (len(data_ch), 100, 100),
    'emb_shapes': ((25, 25), (4, 4)),
    'output_shape': (len(data_ch), 100, 100),
    'fc_output_idx': [2],
    'vq_args': {'num_embeddings': 512, 'embedding_dim': 64},
    'num_class': len(datamanager.unique_labels),
    'fc_input_type': 'vqvec',
}
train_args = {
    'lr': 0.0004,
    'max_epoch': 100,
    'reducelr_patience': 4,
    'reducelr_increment': 0.1,
    'earlystop_patience': 12,
}

# log the training and model args 
with open(results_dir / "model_args.json", 'w') as f:
    json.dump(model_args, f)
with open(results_dir / "train_args.json", 'w') as f:
    json.dump(train_args, f)

trainer = CytoselfFullTrainer(train_args, homepath=results_dir, model_args=model_args)
trainer.fit(datamanager, tensorboard_path=tensorboard_path)

# 2.1 Generate training history
plot_history_cytoself(trainer.history, savepath=trainer.savepath_dict['visualization'])

# 2.2 Compare the reconstructed images as a sanity check
img = next(iter(datamanager.test_loader))['image'].detach().cpu().numpy()
torch.cuda.empty_cache()
reconstructed = trainer.infer_reconstruction(img)
fig, ax = plt.subplots(2, len(data_ch), figsize=(5 * len(data_ch), 5), squeeze=False)
for ii, ch in enumerate(data_ch):
    t0 = np.zeros((2 * 100, 5 * 100))
    for i, im in enumerate(img[:10, ii, ...]):
        i0, i1 = np.unravel_index(i, (2, 5))
        t0[i0 * 100 : (i0 + 1) * 100, i1 * 100 : (i1 + 1) * 100] = im
    t1 = np.zeros((2 * 100, 5 * 100))
    for i, im in enumerate(reconstructed[:10, ii, ...]):
        i0, i1 = np.unravel_index(i, (2, 5))
        t1[i0 * 100 : (i0 + 1) * 100, i1 * 100 : (i1 + 1) * 100] = im
    ax[0, ii].imshow(t0, cmap='gray')
    ax[0, ii].axis('off')
    ax[0, ii].set_title('input ' + ch)
    ax[1, ii].imshow(t1, cmap='gray')
    ax[1, ii].axis('off')
    ax[1, ii].set_title('output ' + ch)
fig.tight_layout()
fig.show()
fig.savefig(join(trainer.savepath_dict['visualization'], 'reconstructed_images.png'), dpi=300)


# 3. Analyze embeddings
analysis = AnalysisOpenCell(datamanager, trainer)

# 3.1 Generate bi-clustering heatmap
analysis.plot_clustermap(num_workers=4)

# 3.2 Generate feature spectrum
vqindhist1 = trainer.infer_embeddings(img, 'vqindhist1')
ft_spectrum = analysis.compute_feature_spectrum(vqindhist1)

x_max = ft_spectrum.shape[1] + 1
x_ticks = np.arange(0, x_max, 50)
fig, ax = plt.subplots(figsize=(10, 3))
ax.stairs(ft_spectrum[0], np.arange(x_max), fill=True)
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('Feature index')
ax.set_ylabel('Counts')
ax.set_xlim([0, x_max])
ax.set_xticks(x_ticks, analysis.feature_spectrum_indices[x_ticks])
fig.tight_layout()
fig.show()
fig.savefig(join(analysis.savepath_dict['feature_spectra_figures'], 'feature_spectrum.png'), dpi=300)


# 3.3 Plot UMAP
umap_data = analysis.plot_umap_of_embedding_vector(
    data_loader=datamanager.test_loader,
    group_col=2,
    output_layer=f'{model_args["fc_input_type"]}2',
    title=f'UMAP {model_args["fc_input_type"]}2',
    xlabel='UMAP1',
    ylabel='UMAP2',
    s=0.3,
    alpha=0.5,
    show_legend=True,
)
