from typing import Optional


class DataManagerBase:
    """
    Abstract class for DataManager
    """

    def __init__(self, basepath: Optional[str], data_split: tuple, shuffle_seed: int, num_workers: int, **kwargs):
        self.basepath = basepath
        self.data_split = data_split
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

    def const_dataloader(self, *args, **kwargs):
        """
        Construct DataLoader objects.
        """
        pass
