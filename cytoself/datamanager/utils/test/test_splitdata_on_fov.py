from math import ceil, floor

import numpy as np

from cytoself.datamanager.utils.splitdata_on_fov import single_proc, splitdata_on_fov


def gen_label(n_fovs: int = 40):
    fov_dict = {f'fov{i}': np.random.randint(10, 40) for i in range(n_fovs)}
    label = []
    for key, val in fov_dict.items():
        lab = np.zeros((val, 2), dtype=object)
        lab[:, 0] = np.random.randint(3)
        lab[:, 1] = key
        label.append(lab)
    return np.vstack(label)


test_label = gen_label()
data_len = len(test_label)


def assert_split_ratio(index_list, split_perc):
    data_list = []
    for d, s in zip(index_list, split_perc):
        data = test_label[d]
        data_list.append(data)
        assert (
            min(1, floor(data_len * s * 0.7)) <= len(data) <= ceil(data_len * s * 1.5)
        ), 'Split ratio deviates too far.'
    return data_list


def assert_no_union(data_list):
    assert not any(
        np.isin(data_list[0][:, 1], data_list[1][:, 1])
    ), 'Duplicated FOVs found between train & validation sets.'
    assert not any(np.isin(data_list[0][:, 1], data_list[2][:, 1])), 'Duplicated FOVs found between train & test sets.'
    assert not any(
        np.isin(data_list[2][:, 1], data_list[1][:, 1])
    ), 'Duplicated FOVs found between test & validation sets.'


def test_single_proc():
    split_perc = (0.8, 0.1, 0.1)
    out = single_proc(test_label, split_perc, fovpath_idx=1)
    data_list = assert_split_ratio(out, split_perc)
    assert_no_union(data_list)


def test_splitdata_on_fov_single():
    split_perc = (0.8, 0.1, 0.1)
    out = splitdata_on_fov(test_label, split_perc, cellline_id_idx=0, fovpath_idx=1, num_workers=1)
    data_list = assert_split_ratio(out, split_perc)
    assert_no_union(data_list)


def test_splitdata_on_fov_multi():
    split_perc = (0.8, 0.1, 0.1)
    out = splitdata_on_fov(test_label, split_perc, cellline_id_idx=0, fovpath_idx=1, num_workers=2)
    data_list = assert_split_ratio(out, split_perc)
    assert_no_union(data_list)


# TODO: Implement cumsum to allow split_perc to have 0.
# def test_splitdata_on_fov_noval():
#     split_perc = (0.8, 0.2, 0)
#     out = splitdata_on_fov(test_label, split_perc, cellline_id_idx=0, fovpath_idx=1, num_workers=1)
#     data_list = assert_split_ratio(out, split_perc)
#     assert_no_union(data_list)
