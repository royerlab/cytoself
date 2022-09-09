from copy import copy
from typing import Optional, Sequence, Union, List
from os.path import join, basename, dirname
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from numpy.typing import ArrayLike
from numpy.distutils.misc_util import is_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

from cytoself.datamanager.utils.splitdata_on_fov import splitdata_on_fov
from cytoself.datamanager.base import DataManagerBase
from cytoself.datamanager.preloaded_dataset import PreloadedDataset


class DataManagerOpenCell(DataManagerBase):
    """
    Manages training, validation and test data for OpenCell data.
    """

    def __init__(
        self,
        basepath: str,
        channel_list: List,
        data_split: tuple = (0.82, 0.098, 0.082),
        label_col: int = 0,
        fov_col: Optional[int] = -1,
        shuffle_seed: int = 1,
        intensity_adjustment: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        basepath : str
            Base path where all numpy files are stored
        channel_list : Sequence of str
            The sequence of channel to be loaded
        data_split : tuple
            Data split ratio
        label_col : int
            Column index to identify proteins
        fov_col : int
            Column index to identify FOVs
        shuffle_seed : int
            Rnadom seed to shuffle data
        intensity_adjustment : dict
            Intensity adjustment for each channel.
        """
        super().__init__(basepath=basepath, data_split=data_split, shuffle_seed=shuffle_seed)
        self.label_col = label_col
        self.fov_col = fov_col
        self.unique_labels = None
        self.train_variance = None
        self.val_variance = None
        self.test_variance = None

        # Make sure label is in the list as data splitting depends on label information.
        if 'label' not in channel_list:
            channel_list = copy(channel_list)
            channel_list.append('label')
        self.channel_list = channel_list

        # Intensity adjustment is needed when using nucdist.
        intensity_adjustment_default = {'pro': 1, 'nuc': 1, 'nucdist': 0.01}
        if isinstance(intensity_adjustment, dict):
            self.intensity_adjustment = intensity_adjustment
        else:
            self.intensity_adjustment = {}
        for key, val in intensity_adjustment_default.items():
            if key not in self.intensity_adjustment:
                self.intensity_adjustment[key] = val

    def determine_load_paths(
        self,
        labels_toload: Sequence[str] = None,
        labels_tohold: Sequence[str] = None,
        num_labels: Optional[int] = None,
        label_name_position: int = -2,
        suffix: Union[str, Sequence] = 'label',
    ):
        """
        Reorganizes the loading DataFrame with designated labels.
        Loading list can be reorganized by plugging label names exclusively to be included or excluded as well as
        the total number of labels.

        Parameters
        ----------
        labels_toload : Sequence of str
            Label names to be loaded
        labels_tohold : Sequence of str
            Label names to be excluded
        num_labels : int
            Number of labels to be loaded
        label_name_position : int
            Relative position of label name from suffix in the npy file name
        suffix : str or sequence of str
            The file suffix to be loaded

        Returns
        -------
        A DataFrame containing labels and file paths to be loaded

        """
        file_df = get_file_df(self.basepath, suffix)
        if labels_toload:
            ind0 = (
                file_df.iloc[:, 0]
                .str.split('/', expand=True)
                .iloc[:, -1]
                .str.split('_', expand=True)
                .iloc[:, label_name_position]
                .isin(labels_toload)
            )
            file_df = file_df[ind0]
        if labels_tohold:
            ind0 = (
                file_df.iloc[:, 0]
                .str.split('/', expand=True)
                .iloc[:, -1]
                .str.split('_', expand=True)
                .iloc[:, label_name_position]
                .isin(labels_tohold)
            )
            file_df = file_df[~ind0]
        return file_df.iloc[:num_labels]

    def _load_data_multi(self, df_toload: DataFrame):
        """
        Load numpy files with multiprocessing

        Parameters
        ----------
        df_toload : DataFrame
            DataFrame of labels and file paths to be loaded

        Returns
        -------
        Image and label data in a tuple of numpy arrays

        """
        image_all, label_all = [], []
        for ch in self.channel_list:
            print(f'Loading {ch} data...')
            results = Parallel(n_jobs=self.num_workers)(
                delayed(np.load)(row[ch], allow_pickle=ch == 'label')
                for _, row in tqdm(df_toload.iterrows(), total=len(df_toload))
            )
            if ch == 'label':
                label_all = np.vstack(results)
            else:
                if results[0].ndim == 3:
                    d = np.vstack(results)[..., np.newaxis]
                else:
                    d = np.vstack(results)
                if ch in self.intensity_adjustment:
                    d *= self.intensity_adjustment[ch]
                else:
                    print('Channel not found in intensity balance.')
                image_all.append(d)
        if len(image_all) > 0:
            image_all = np.concatenate(image_all, axis=-1)
        else:
            print('No image data was loaded.')
        return image_all, label_all

    def split_data(self, label_data: ArrayLike):
        """
        Split data into train, validation and test sets.

        Parameters
        ----------
        label_data : Numpy array
            Label information in a numpy array

        Returns
        -------
        Indices of training, validation and test sets in a tuple of numpy arrays

        """
        # Split data
        print('Splitting data...')
        if self.fov_col is None:
            np.random.seed(self.shuffle_seed)
            ind = np.random.choice(len(label_data), size=len(label_data), replace=False)
            split_ind = list(np.cumsum([int(len(label_data) * i) for i in self.data_split[:-1]]))
            train_ind = ind[0 : split_ind[0]]
            val_ind = ind[split_ind[0] : split_ind[1]]
            test_ind = ind[split_ind[1] : len(label_data)]
        else:
            train_ind, val_ind, test_ind = splitdata_on_fov(
                label_data,
                split_perc=self.data_split,
                cellline_id_idx=self.label_col,
                fovpath_idx=self.fov_col,
                num_workers=self.num_workers,
                shuffle_seed=self.shuffle_seed,
            )
        return train_ind, val_ind, test_ind

    def const_label_book(self, label_data: ArrayLike):
        """
        Constructs label book for classification tasks
        This function creates an array of unique labels whose index becomes the label for the classification tasks.

        Parameters
        ----------
        label_data : ArrayLike
            Label array

        """
        self.unique_labels = np.unique(label_data[:, self.label_col])

    def const_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[Sequence] = (
            transforms.RandomApply(
                [
                    lambda x: transforms.functional.rotate(x, 0),
                    lambda x: transforms.functional.rotate(x, 90),
                    lambda x: transforms.functional.rotate(x, 180),
                    lambda x: transforms.functional.rotate(x, 270),
                ]
            ),
            # transforms.RandomRotation(180, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        ),
        labels_toload: Optional[Sequence[str]] = None,
        labels_tohold: Optional[Sequence[str]] = None,
        num_labels: Optional[int] = None,
        label_format: Optional[str] = 'index',
        label_name_position: int = -2,
        shuffle: bool = True,
        shuffle_test: bool = False,
        **kwargs,
    ):
        """
        Loads and splits data into training, validation and test data, followed by DataLoader construction.

        Parameters
        ----------
        batch_size : int
            Batch size
        num_workers : int
            Number of workers for multiprocessing
        transform : Callable
            Data augmentation functions
        labels_toload : Sequence of str
            Label names to be loaded
        labels_tohold : Sequence of str
            Label names to be excluded
        num_labels : int
            Number of labels to be loaded
        label_format : str or None
            Format of converted label: onehot, index or None (i.e. no conversion)
        label_name_position : int
            Relative position of label name from suffix in the npy file name
        shuffle : bool
            Shuffle train & val batches if True
        shuffle_test : bool
            Shuffle test batch if True.

        """
        self.num_workers = num_workers
        transform_all = transforms.Compose([torch.from_numpy] + ([] if transform is None else list(transform)))
        # Determine which npy files to load.
        df_toload = self.determine_load_paths(
            labels_toload, labels_tohold, num_labels, label_name_position, self.channel_list
        )

        # Load data
        image_all, label_all = self._load_data_multi(df_toload)
        if len(image_all) > 0:
            image_all = np.moveaxis(image_all, np.argmin(image_all.shape), 1)

        # Get unique labels
        self.const_label_book(label_all)

        # Split data
        train_ind, val_ind, test_ind = self.split_data(label_all)

        if len(train_ind) > 0:
            train_label = label_all[train_ind]
            if len(image_all) > 0:
                train_data = image_all[train_ind]
            else:
                train_data = []
            train_dataset = PreloadedDataset(
                train_label, train_data, transform_all, self.unique_labels, label_format, self.label_col
            )
            _assert_dtype(train_dataset.label, train_dataset.label_format)
            self.train_loader = DataLoader(
                train_dataset, batch_size, shuffle=shuffle, num_workers=self.num_workers, **kwargs
            )
            print('Computing variance of training data...')
            self.train_variance = np.var(train_data).item()
        if len(val_ind) > 0:
            val_label = label_all[val_ind]
            if len(image_all) > 0:
                val_data = image_all[val_ind]
            else:
                val_data = []
            val_dataset = PreloadedDataset(
                val_label, val_data, transform_all, self.unique_labels, label_format, self.label_col
            )
            _assert_dtype(val_dataset.label, val_dataset.label_format)
            self.val_loader = DataLoader(
                val_dataset, batch_size, shuffle=shuffle, num_workers=self.num_workers, **kwargs
            )
            print('Computing variance of validation data...')
            self.val_variance = np.var(val_data).item()
        if len(test_ind) > 0:
            test_label = label_all[test_ind]
            if len(image_all) > 0:
                test_data = image_all[test_ind]
            else:
                test_data = []
            test_dataset = PreloadedDataset(
                test_label, test_data, None, self.unique_labels, label_format, self.label_col
            )
            _assert_dtype(test_dataset.label, test_dataset.label_format)
            self.test_loader = DataLoader(
                test_dataset, batch_size, shuffle=shuffle_test, num_workers=self.num_workers, **kwargs
            )
            print('Computing variance of test data...')
            self.test_variance = np.var(test_data).item()

    @staticmethod
    def download_sample_data(output: Optional[str] = 'sample_data'):
        """
        Download sample data

        Parameters
        ----------
        output : str
            Destination path
        ----------
        """
        import gdown

        gdown.download_folder(
            url='https://drive.google.com/drive/folders/1tCgFlcyBg8p7241eowlDi7EsUmiR42h9',
            output=output,
            quiet=False,
        )


def _assert_dtype(label, label_format):
    """
    Asserts dtype for DataLoader

    Parameters
    ----------
    label : numpy array
        Label data
    label_format : str
        label_format argument in const_dataset

    Returns
    -------
    None

    """
    if (
        label_format is None
        and isinstance(label, np.ndarray)
        and not (np.issubdtype(label.dtype, np.number) or np.issubdtype(label.dtype, bool))
    ):
        raise TypeError(f'label must be numerical to use dataloader, instead {label.dtype} is given.')


def get_file_df(basepath: str, suffix: Union[str, Sequence] = 'label', extension: str = 'npy'):
    """
    Creates a DataFrame of data paths.

    Parameters
    ----------
    basepath : str
        The base path that contains all npy files
    suffix : str or sequence of str
        The file suffix used to create the DataFrame
    extension : str
        The file extension used to create the DataFrame

    Returns
    -------
    A DataFrame of data paths to load.

    """
    df = pd.DataFrame()
    if isinstance(suffix, str):
        df[suffix] = sorted(glob(join(basepath, '*_' + suffix + '.' + extension)))
    elif is_sequence(suffix):
        filelist = sorted(glob(join(basepath, '*_' + suffix[0] + '.' + extension)))
        df[suffix[0]] = filelist
        for sf in suffix[1:]:
            flist = []
            for p in filelist:
                flist.append(join(dirname(p), '_'.join(basename(p).split('_')[:-1]) + '_' + sf + '.' + extension))
            df[sf] = flist
    else:
        raise TypeError('Only str or list is accepted for suffix.')
    return df
