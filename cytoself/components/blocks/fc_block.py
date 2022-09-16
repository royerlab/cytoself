import math

import torch
from torch import nn

from cytoself.components.utils.activation_selecter import act_layer


class FCblock(nn.Module):
    """
    A block of fc layers
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features,
        num_layers: int,
        dropout_rate: float = 0.5,
        act: str = 'relu',
        last_activation: str = 'softmax',
    ):
        """
        A block of fully connected layers

        Parameters
        ----------
        in_channels : int
            Input channels or features
        out_channels : int
            Output channels or features
        num_features : int
            Number of features in the intermediate layers
        num_layers : int
            Number of fc layers
        dropout_rate : float
            Dropout rate
        act : str
            Activation name for the intermediate layers
        last_activation : str
            Activation name as the output layer
        """
        super().__init__()
        self.last_activation = last_activation
        self.fc_list = nn.ModuleList()
        for i in range(num_layers):
            self.fc_list.append(nn.Dropout(dropout_rate, inplace=False))
            self.fc_list.append(
                nn.Linear(in_channels if i == 0 else num_features, num_features if i < num_layers - 1 else out_channels)
            )
            if i < num_layers - 1:
                self.fc_list.append(act_layer(act))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if not torch.is_floating_point(x):
            x = x.type(torch.float32)
        for lyr in self.fc_list:
            x = lyr(x)
        return x
