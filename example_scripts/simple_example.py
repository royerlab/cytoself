from cytoself.datamanager.opencell import DataManagerOpenCell
from cytoself.trainer.cytoselflite_trainer import CytoselfLiteTrainer
from cytoself.analysis.analysis_opencell import AnalysisOpenCell


# 1. Prepare Data
datapath = 'sample_data'  # path to download sample data
DataManagerOpenCell.download_sample_data(datapath)  # donwload data
datamanager = DataManagerOpenCell(datapath, ['pro'], fov_col=None)
datamanager.const_dataloader(batch_size=32, label_name_position=1)


# 2. Create and train a cytoself model
model_args = {
    'input_shape': (1, 100, 100),
    'emb_shapes': ((64, 25, 25), (64, 4, 4)),
    'output_shape': (1, 100, 100),
    'vq_args': {'num_embeddings': 512},
    'num_class': len(datamanager.unique_labels),
}
train_args = {
    'lr': 1e-3,
    'max_epoch': 10,
    'reducelr_patience': 3,
    'reducelr_increment': 0.1,
    'earlystop_patience': 6,
}
trainer = CytoselfLiteTrainer(train_args, homepath='demo_output', model_args=model_args)
trainer.fit(datamanager, tensorboard_path='tb_logs')


# 3. Plot UMAP
analysis = AnalysisOpenCell(datamanager, trainer)
umap_data = analysis.plot_umap_of_embedding_vector(
    data_loader=datamanager.test_loader,
    group_col=2,
    output_layer='vqvec2',
    title='UMAP vqvec2',
    xlabel='UMAP1',
    ylabel='UMAP2',
    s=2,
    alpha=1,
    show_legend=True,
)
