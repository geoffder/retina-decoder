# import torch
from torch import nn
# import torch.nn.functional as F


class Permuter(nn.Module):
    "Returns new view of input tensor with dims swapped around."
    def __init__(self, new_order):
        super(Permuter, self).__init__()
        self.new_order = new_order  # tuple

    def forward(self, X):
        return X.permute(*self.new_order)


def make_pool3d_layer(param_dict):
    params = (
        param_dict['kernel'], param_dict.get('stride', param_dict['kernel']),
        param_dict.get('padding', 0)
    )
    if param_dict.get('op', 'avg') == 'max':
        return nn.MaxPool3d(*params)
    else:
        return nn.AvgPool3d(*params)
