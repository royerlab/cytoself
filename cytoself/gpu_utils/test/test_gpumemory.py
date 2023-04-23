import torch

from cytoself.gpu_utils.gpumemory import getbestgpu, getfreegpumem, gpuinfo


def test_getbestgpu_cpu():
    if torch.cuda.is_available():
        assert len(gpuinfo(0)) > 0
        assert isinstance(getfreegpumem(0), int)
        assert isinstance(getbestgpu(), int)
    else:
        assert len(gpuinfo(0)) == 0
        assert getfreegpumem(0) == -1
        assert getbestgpu() == -1
