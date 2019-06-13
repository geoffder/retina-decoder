# import torch
from torch import nn
# import torch.nn.functional as F
from temporal_convolution import CausalPool3d


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
        if not param_dict.get('causal', False):
            return nn.MaxPool3d(*params)
        else:
            return CausalPool3d('max', *params[:-1])
    else:
        if not param_dict.get('causal', False):
            return nn.AvgPool3d(*params)
        else:
            return CausalPool3d('avg', *params[:-1])
