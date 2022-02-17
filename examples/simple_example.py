import numpy as np
import pandas as pd
import gdown
from cytoself.data_loader.data_manager import DataManager
from cytoself.models import CytoselfFullModel
from cytoself.data_loader.data_generator import image_and_label_generator
from cytoself.analysis.analytics import Analytics

"""
A simple demo. 
"""

#%% Download example data ================================================================================
print("Downloading data...")
# Download model weights
gdown.download(
    'https://drive.google.com/uc?id=1gkiEMKdadOel4Xh6KoS2U603JTkZhgDw',
    'pretrained_model.h5',
    quiet=False
)
# Download label data
gdown.download(
    'https://drive.google.com/uc?id=16-0bhKrUMbZ0DSz768Z_q13yNivHyfVO',
    'example_label.npy',
    quiet=False
)
# Download image data
gdown.download(
    'https://drive.google.com/uc?id=1znRLbYJJqd11Zqv-5_yUmNjarKcwIWMg',
    'example_image.npy',
    quiet=False
)
# Download localization table
gdown.download(
    'https://drive.google.com/uc?id=1RM654Qavcy8gG5uy3mCzi8EsOT_xOlVd',
    'protein_uniloc.csv',
    quiet=False
)
# Download dendrogram index to plot feature spectrum
gdown.download(
    'https://drive.google.com/uc?id=1WrxhGsSzivZVAlL_K2FLVsRmHrsfhyrI',
    'dgram_index1.npy',
    quiet=False
)


#%% Load example data =====================================================================================
print("Loading data...")
image_data = np.load('example_image.npy')
label_data = np.load('example_label.npy', allow_pickle=True)
gt_table = pd.read_csv('protein_uniloc.csv')

# The image data has 3 channels which are protein label, nucleus and nucleus distance.
# In this example we only use protein label and nucleus distance channels.
image_data = image_data[:, ..., [0, 2]]

# Make sure that the label data has 2 dimensions.
label_data = label_data.reshape(-1, 1)


#%% Creat model and DataManager instance ===================================================================
print("Constructing model and DataManager...")
model = CytoselfFullModel(input_image_shape=[100, 100, 2], num_fc_output_classes=len(np.unique(label_data)))
data_manager = DataManager(
        train_data=image_data[:100],
        val_data=image_data[100:200],
        test_data=image_data[200:],
        train_label=label_data[:100],
        val_label=label_data[100:200],
        test_label=label_data[200:],
)
# Note. The data split here is only to provide an example of how to run cytoself. Please make sure the data is
# split properly when you train your read data.

# Compile the model with data_manager
model.compile_with_datamanager(data_manager)


#%% Train and load the model ===============================================================================
print("Training model...")
# This is just a demonstration of how to train a model, so the training will end in a short time.
model.train_with_datamanager(data_manager, batch_size=64, max_epoch=1)

# To pretend we have trained the model well, let's load a pretrained model.
model.load_model('pretrained_model.h5')


#%% Analyze results ========================================================================================
print("Analyzing embedding...")
# Create a Analytics instance to visualize analysis results
analytics = Analytics(model, data_manager, gt_table)

# Plot UMAPs of representation vectors
analytics.calc_plot_umaps_gt("vec", titles="Unique localization")

# Plot clustermaps
analytics.plot_clustermaps()

# Load dendrogram index and plot feature spectrum
analytics.load_dendrogram_index('dgram_index1.npy')
analytics.plot_feature_spectrum_from_image(image_data[:1])
