from typing import Optional


class DataManagerBase:
    """
    Abstract class for DataManager
    """

    def __init__(self, basepath: Optional[str], data_split: tuple, shuffle_seed: int, **kwargs):
        self.basepath = basepath
        self.data_split = data_split
        self.shuffle_seed = shuffle_seed
        self.num_workers = 1
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def const_dataloader(self, *args, **kwargs):
        """
        Construct DataLoader objects.
        """
        pass
