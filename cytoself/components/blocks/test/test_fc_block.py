import torch
from torch import nn

from cytoself.components.blocks.fc_block import FCblock


def test_FCblock_train():
    fc_block = FCblock(10, 3, 5, 2)
    for i, lyr in enumerate([nn.Dropout, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear]):
        assert isinstance(fc_block.fc_list[i], lyr)
    out = fc_block(torch.randn((4, 10)))
    assert out.shape == (4, 3)


def test_FCblock_infer():
    fc_block = FCblock(10, 3, 5, 2)
    fc_block.eval()
    out = fc_block(torch.randn((4, 10)))
    assert out.shape == (4, 3)
