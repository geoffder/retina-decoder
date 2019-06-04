import torch
from torch import nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """
    Implementation of an inception style convolutional block. For simplicity,
    this block will follow a stereotyped structure, but the number of
    features maps at each stage are specified as follows:
        [in features, bottleneck features, out features]
    Note that since each of the 4 branches within the block are concatenated,
    the number of features maps of the output of this block is the specified
    'out features' * 4.
    """
    def __init__(self, features):
        super(InceptionBlock, self).__init__()
        self.features = features  # [in, bottleneck, out]
        self.build()

    def build(self):
        # branch 1: (1x1) conv only
        self.br1_conv1 = nn.Conv2d(
                self.features[0], self.features[2], (1, 1), stride=1,
                padding=(0, 0), bias=False
        )
        self.br1_bnorm1 = nn.BatchNorm2d(self.features[2])

        # branch 2: (1x1) conv bottleneck -> (3x3) conv
        self.br2_conv1 = nn.Conv2d(
                self.features[0], self.features[1], (1, 1), stride=1,
                padding=(0, 0), bias=False
        )
        self.br2_bnorm1 = nn.BatchNorm2d(self.features[1])
        self.br2_conv3 = nn.Conv2d(
                self.features[1], self.features[2], (3, 3), stride=1,
                padding=(1, 1), bias=False
        )
        self.br2_bnorm3 = nn.BatchNorm2d(self.features[2])

        # branch 3: (1x1) conv bottleneck -> (5x5) conv
        self.br3_conv1 = nn.Conv2d(
                self.features[0], self.features[1], (1, 1), stride=1,
                padding=(0, 0), bias=False
        )
        self.br3_bnorm1 = nn.BatchNorm2d(self.features[1])
        self.br3_conv5 = nn.Conv2d(
                self.features[1], self.features[2], (5, 5), stride=1,
                padding=(2, 2), bias=False
        )
        self.br3_bnorm5 = nn.BatchNorm2d(self.features[2])

        # branch 4: (3x3) avg pool (stride=1) -> (1x1) conv
        self.br4_avgpool = nn.AvgPool2d(3, stride=1, padding=1)
        self.br4_conv1 = nn.Conv2d(
                self.features[0], self.features[2], (1, 1), stride=1,
                padding=(0, 0), bias=False
        )
        self.br4_bnorm1 = nn.BatchNorm2d(self.features[2])

    def forward(self, X):
        X1 = self.br1_bnorm1(self.br1_conv1(X))
        X2 = self.br2_bnorm3(self.br2_conv3(
            F.relu(self.br2_bnorm1(self.br2_conv1(X)))
        ))
        X3 = self.br3_bnorm5(self.br3_conv5(
            F.relu(self.br3_bnorm1(self.br3_conv1(X)))
        ))
        X4 = self.br4_bnorm1(self.br4_conv1(self.br4_avgpool(X)))
        # concatenate on feature-map dimension (N x [C] x H x W)
        X = torch.cat([X1, X2, X3, X4], dim=1)
        return F.relu(X)
