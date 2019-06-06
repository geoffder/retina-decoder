import torch
from torch import nn
from torch.nn.utils import weight_norm

"""
Inspired by https://github.com/locuslab/TCN/blob/master/TCN/tcn.py#L53.
TemporalBlock and Chomp Adapted to work with 3d convolutions (video data).
"""


class Chomp3d(nn.Module):
    "Expects (_, _, T, H, W) input. Bites off the end of the sequence dim T."
    def __init__(self, chomp_size):
        super(Chomp3d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, X):
        return X[:, :, :-self.chomp_size, :, :].contiguous()


class TemporalBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, dilation, padding, groups=1, dropout=0,
                 activation=nn.ReLU):
        super(TemporalBlock3d, self).__init__()

        # in_channels -> out_channels convolution, activation, and dropout
        self.conv1 = weight_norm(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )
        )
        self.chomp1 = Chomp3d(padding[0])
        self.activation1 = activation()
        self.dropout1 = nn.Dropout(dropout)

        # out_channels -> out_channels convolution, activation, and dropout
        self.conv2 = weight_norm(
            nn.Conv3d(
                out_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )
        )
        self.chomp2 = Chomp3d(padding[0])
        self.activation2 = activation()
        self.dropout2 = nn.Dropout(dropout)

        # package the main block as a sequential model
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.activation1, self.dropout1,
            self.conv2, self.chomp2, self.activation2, self.dropout2
        )

        # 1x1x1 kernel convolution to adjust dimensionality of skip connection
        self.downsample = nn.Conv3d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None
        self.activation = activation()

        self.init_weights()

    def init_weights(self):
        "Re-initialize weight standard deviation; 0.1 -> .01"
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, X):
        "Residual connection dimensionality reduced/increased to match main."
        out = self.net(X)
        res = X if self.downsample is None else self.downsample(X)
        return self.activation(out + res)


class TemporalConv3dStack(nn.Module):
    def __init__(self, in_channels, block_channels, kernel_size=(2, 1, 1),
                 space_dilation=1, groups=1, dropout=0, activation=nn.ReLU):
        super(TemporalConv3dStack, self).__init__()
        blocks = []
        for i, out_channels in enumerate(block_channels):
            time_dilation = 2**i
            padding = (
                (kernel_size[0]-1)*time_dilation,  # causal padding
                (kernel_size[1]*space_dilation - 1)//2,
                (kernel_size[2]*space_dilation - 1)//2
            )
            blocks.append(
                TemporalBlock3d(
                    in_channels, out_channels, kernel_size, stride=1,
                    dilation=(time_dilation, space_dilation, space_dilation),
                    padding=padding, groups=1, dropout=dropout
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # generate random data and build Convolutional RNN cells
    data = torch.randn(5, 15, 60, 30, 30).to(device)  # (N, C, T, H, W)
    tcnn = TemporalConv3dStack(
        15,  # input channels
        [30, 60, 30],  # channels for each block
        kernel_size=(2, 1, 1),
        space_dilation=1,
        groups=1,
        dropout=0,
        activation=nn.Tanh
    ).to(device)

    # run data through Temporal Convolution network
    out = tcnn(data)

    # convert data and outputs into numpy
    data = data.detach().cpu().numpy()
    out = out.detach().cpu().numpy()

    # input
    print('input dimensions (N, C, T, H, W)')
    print('input shape:', data.shape, end='\n\n')
    # sequence outputs
    print('output dimensions (N, C, T, H, W)')
    print('output shape:', out.shape)
