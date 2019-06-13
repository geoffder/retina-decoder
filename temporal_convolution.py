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
        # Select forward() method. (Do nothing if chomp_size is 0)
        self.forward = self.chomp if chomp_size else self.skip

    def chomp(self, X):
        "Usual forward operation."
        return X[:, :, :-self.chomp_size, :, :].contiguous()

    def skip(self, X):
        "Don't try to chomp, [:-0] results in a terminal error."
        return X


class TemporalBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation, padding, groups=1, dropout=0, activation=nn.ReLU):
        super(TemporalBlock3d, self).__init__()

        # in_channels -> out_channels convolution, activation, and dropout
        conv1 = weight_norm(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )
        )
        chomp1 = Chomp3d(padding[0])
        activation1 = activation()
        dropout1 = nn.Dropout(dropout)

        # out_channels -> out_channels convolution, activation, and dropout
        conv2 = weight_norm(
            nn.Conv3d(
                out_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )
        )
        chomp2 = Chomp3d(padding[0])
        activation2 = activation()
        dropout2 = nn.Dropout(dropout)

        # package the main block as a sequential model
        self.main_branch = nn.Sequential(
            conv1, chomp1, activation1, dropout1,
            conv2, chomp2, activation2, dropout2
        )

        # 1x1x1 kernel convolution to adjust dimensionality of skip connection
        self.downsample = nn.Conv3d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None
        self.activation = activation()

        self.init_weights()

    def init_weights(self):
        "Re-initialize weight standard deviation; 0.1 -> .01"
        self.main_branch[0].weight.data.normal_(0, 0.01)  # conv1
        self.main_branch[4].weight.data.normal_(0, 0.01)  # conv2
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, X):
        "Residual connection dimensionality reduced/increased to match main."
        out = self.main_branch(X)
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

    def forward(self, X):
        return self.network(X)


class CausalTranspose3d(nn.Module):
    '''
    ConvTranspose outputs are GROWN by the kernel rather than shrank, and the
    padding parameter serves to cancel it out (rather than add to it). Thus
    to achieve causality in the temporal dimension, depth "anti-padding" is
    set to 0, and the implicit transpose convolution padding is "chomped" off.
    '''
    def __init__(self, in_channels, out_channels, kernel, stride, groups=1,
                 bias=True, dilation=(1, 1, 1)):
        super(CausalTranspose3d, self).__init__()

        # unpack tuples for padding calculations
        d, h, w = kernel
        st_d, st_h, st_w = stride
        dil_d, dil_h, dil_w = dilation

        # calculate 'same' padding in spatial dimensions
        padding = (0, h*dil_h//2, w*dil_w//2)
        # asymmetrical padding to achieve 'same' dimensions despite upsampling
        out_padding = (st_d//2, st_h//2, st_w//2)

        self.network = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, (d, h, w), (st_d, st_h, st_w),
                padding, out_padding, groups, bias, dilation,
            ),
            # remove all implicit T padding from the end (therefore causal)
            Chomp3d((d-1)*dil_d)
        )

    def forward(self, X):
        return self.network(X)


class CausalPool3d(nn.Module):
    '''
    Provides causal padding for 3d pooling operations (op='avg' or 'max').
    No padding in spatial, usual pooling convention is only deviated from in
    the temporal dimension.
    '''
    def __init__(self, op, kernel, stride=None):
        super(CausalPool3d, self).__init__()

        stride = kernel if stride is None else stride
        padding = (kernel[0]-1, 0, 0)  # Causal padding in time.

        if op == 'avg':
            pool = nn.AvgPool3d(
                kernel, stride, padding, count_include_pad=False
            )
        elif op == 'max':
            pool = nn.MaxPool3d(kernel, stride, padding)

        self.network = nn.Sequential(pool, Chomp3d(padding[0]))

    def forward(self, X):
        return self.network(X)


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

    # causal pooling
    pool = CausalPool3d('avg', (2, 2, 2)).to(device)
    out = pool(out)

    # convert data and outputs into numpy
    data = data.detach().cpu().numpy()
    out = out.detach().cpu().numpy()

    # input
    print('input dimensions (N, C, T, H, W)')
    print('input shape:', data.shape, end='\n\n')
    # sequence outputs
    print('output dimensions (N, C, T, H, W)')
    print('output shape:', out.shape)
