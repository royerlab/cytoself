import numpy as np

from cytoself.data_loader.data_generator import image_and_label_generator


class DataManager:
    """
    A class object to manage training, validation and test data.
    """

    def __init__(
        self,
        train_data=None,
        val_data=None,
        test_data=None,
        train_label=None,
        val_label=None,
        test_label=None,
    ):
        self.train_data = [] if train_data is None else train_data
        self.val_data = [] if val_data is None else val_data
        self.test_data = [] if test_data is None else test_data
        self.train_label = [] if train_label is None else train_label
        self.val_label = [] if val_label is None else val_label
        self.test_label = [] if test_label is None else test_label
        self.train_label_onehot = []
        self.val_label_onehot = []
        self.test_label_onehot = []
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.uniq_label = None
        self.n_classes = None

    def get_unique_labels(self, col=0):
        """
        Compute unique labels throughout train, validate and test data.
        :param col: column index of label to be converted to onehot vector.
        """
        label_all = np.vstack(
            [
                d[:, col : col + 1]
                for d in [self.train_label, self.val_label, self.test_label]
                if len(d) > 0
            ]
        )
        self.uniq_label = np.unique(label_all)
        self.n_classes = len(self.uniq_label)

    def get_label_onehot(self, col=0):
        """
        Convert label to onehot vector.
        :param col: column index of label to be converted to onehot vector.
        """
        # Get unique label values throughout train, val and test labels.
        self.get_unique_labels(col=col)

        # Convert
        self.train_label_onehot = np.zeros(
            (len(self.train_label), self.n_classes), dtype=np.float32
        )
        self.val_label_onehot = np.zeros(
            (len(self.val_label), self.n_classes), dtype=np.float32
        )
        self.test_label_onehot = np.zeros(
            (len(self.test_label), self.n_classes), dtype=np.float32
        )
        for i, c in enumerate(self.uniq_label):
            self.train_label_onehot[:, i] = self.train_label[:, col] == c
            self.val_label_onehot[:, i] = self.val_label[:, col] == c
            self.test_label_onehot[:, i] = self.test_label[:, col] == c

    def make_generators(self, batch_size, n_label_out=2, col=0):
        """
        Make data generators
        :param batch_size: batch size
        :param n_label_out: number of label output
        :param col: col argument of get_label_onehot
        """
        if (
            len(self.train_label_onehot) == 0
            or len(self.val_label_onehot) == 0
            or len(self.test_label_onehot) == 0
        ):
            self.get_label_onehot(col)
        if len(self.train_data) > 0:
            self.train_generator = image_and_label_generator(
                self.train_data,
                self.train_label_onehot,
                n_label_out=2,
                batch_size=batch_size,
                flip_dimension=True,
                rot90=True,
            )
        else:
            print("train_data not found. train_generator was not created.")
        if len(self.val_data) > 0:
            self.val_generator = image_and_label_generator(
                self.val_data,
                self.val_label_onehot,
                n_label_out=n_label_out,
                batch_size=batch_size,
                flip_dimension=True,
                rot90=True,
            )
        else:
            print("val_data not found. val_generator was not created.")
        if len(self.test_data) > 0:
            self.test_generator = image_and_label_generator(
                self.test_data,
                self.test_label_onehot,
                n_label_out=n_label_out,
                batch_size=batch_size,
                no_repeat=True,
            )
        else:
            print("test_data not found. test_generator was not created.")
