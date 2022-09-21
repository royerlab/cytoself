from os.path import join

import numpy as np

from cytoself.test_util.test_parameters import test_label


def gen_npy(basepath, shape=(10, 10)):
    for i in range(test_label[:, 0].max() + 1):
        ind = test_label[:, 0] == i
        np.save(join(basepath, f'protein{i}_pro.npy'), np.zeros((sum(ind),) + shape, dtype=np.uint8))
        np.save(join(basepath, f'protein{i}_nuc.npy'), np.zeros((sum(ind),) + shape, dtype=np.uint8))
        np.save(join(basepath, f'protein{i}_label.npy'), test_label[ind])
