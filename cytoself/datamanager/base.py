from typing import Optional, Callable
from numpy.typing import ArrayLike
from torch.utils.data import Dataset


class DataManagerBase:
    """
    Abstract class for DataManager
    """

    def __init__(
        self,
        basepath: Optional[str],
        data_split: tuple,
        batch_size: int,
        shuffle_seed: int,
        num_workers: int,
        **kwargs,
    ):
        self.basepath = basepath
        self.data_split = data_split
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.num_workers = num_workers
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def const_dataset(self, *args, **kwargs):
        """
        Construct DataSet objects.
        """
        pass


class PreloadedDataset(Dataset):
    """
    Dataset class for preloaded data
    """

    def __init__(
        self,
        label: ArrayLike,
        data: Optional[ArrayLike] = None,
        transform: Optional[Callable] = None,
        unique_labels: ArrayLike = None,
        label_format: Optional[str] = None,
        label_col: int = 0,
    ):
        """
        Parameters
        ----------
        label : ArrayLike
            Label array
        data : ArrayLike
            Image data
        transform : Callable
            Functions for data augmentation
        unique_labels : ArrayLike
            Array of unique labels, serving as the canonical label book for classification tasks
        label_format : str or None
            Format of converted label: onehot, index or None (i.e. no conversion)
        label_col : int
            Column index to be used for label
        """
        self.data = data
        self.label = label
        self.transform = transform
        self.unique_labels = unique_labels
        self.label_format = label_format
        self.label_col = label_col

    def __len__(self):
        return len(self.label)

    def label_converter(self, label):
        if self.unique_labels is None:
            raise ValueError('unique_labels is empty.')
        else:
            onehot = label[:, None] == self.unique_labels
            if self.label_format == 'onehot':
                return onehot[0]
            else:
                return onehot.argmax(1)

    def __getitem__(self, idx):
        label = self.label[idx].reshape(-1, self.label.shape[1])
        label = label[:, self.label_col]
        if self.label_format:
            label = self.label_converter(label)
        if self.data is None or len(self.data) == 0:
            return {'label': label}
        else:
            data = self.data[idx]
            if self.transform is not None:
                data = self.transform(data)
            return {'image': data, 'label': label}
