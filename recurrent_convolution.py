import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def init_filter(shape):
    "Initialize a conv2d filter. Shape (out_channels, in_channels, H, W)."
    W = torch.empty(*shape).normal_(0, .01)
    return nn.Parameter(W)


class ConvGRUCell(nn.Module):

    def __init__(self, dims, in_kernel, out_kernel, in_channels, out_channels,
                 learn_initial=False, return_hidden=False):
        super(ConvGRUCell, self).__init__()
        self.dims = dims  # input spatial dimentions (H, W)
        self.in_kernel = in_kernel  # 2D input filter kernel shape (H, W)
        self.out_kernel = out_kernel  # 2D hidden filter kernel shape (H, W)
        self.in_channels = in_channels  # number input feature maps
        self.out_channels = out_channels  # number hidden feature maps
        self.in_shape = (out_channels, in_channels, *in_kernel)
        self.out_shape = (out_channels, out_channels, *out_kernel)
        self.in_pad = (in_kernel[0]//2, in_kernel[1]//2)  # conv padding
        self.out_pad = (out_kernel[0]//2, out_kernel[1]//2)  # conv padding
        self.learn_initial = learn_initial  # trainable initial state
        self.return_hidden = return_hidden  # returns (output, hidden)
        self.build()

    def build(self):
        # input weight (transforms X before entering the hidden recurrence)
        self.Wxh = init_filter(self.in_shape)
        self.bxh = nn.Parameter(torch.zeros(self.out_channels).float())
        # hidden weight and bias
        self.Whh = init_filter(self.out_shape)
        self.bhh = nn.Parameter(torch.zeros(self.out_channels).float())
        # update gate weights
        self.Wxz = init_filter(self.in_shape)
        self.bxz = nn.Parameter(torch.zeros(self.out_channels).float())
        self.Whz = init_filter(self.out_shape)
        self.bhz = nn.Parameter(torch.zeros(self.out_channels).float())
        # reset gate weights
        self.Wxr = init_filter(self.in_shape)
        self.bxr = nn.Parameter(torch.zeros(self.out_channels).float())
        self.Whr = init_filter(self.out_shape)
        self.bhr = nn.Parameter(torch.zeros(self.out_channels).float())
        # initial hidden repesentation
        self.h0 = nn.Parameter(
            torch.zeros(self.out_channels, *self.dims).float(),
            requires_grad=self.learn_initial
        )

    def forward(self, X):
        """
        Input shape is (T, batch, C, H, W). Conv2d takes (batch, C, H, W).
        Loop over T and collect output in list, then stack back up on time
        dimension. T is not ragged, videos are all the same duration.
        """
        out = []
        # repeat initial hidden state for each sample
        hidden = self.h0.repeat(X.shape[1], 1, 1, 1)
        for frame in X:
            # calculate gates
            reset_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxr, bias=self.bxr, padding=self.in_pad)
                + F.conv2d(
                    hidden, self.Whr, bias=self.bhr, padding=self.out_pad
                )
            )
            update_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxz, bias=self.bxz, padding=self.in_pad)
                + F.conv2d(
                    hidden, self.Whz, bias=self.bhz, padding=self.out_pad
                )
            )
            # update hidden representation
            h_hat = torch.tanh(
                F.conv2d(frame, self.Wxh, bias=self.bxh, padding=self.in_pad)
                + F.conv2d(
                    reset_gate*hidden, self.Whh, bias=self.bhh,
                    padding=self.out_pad
                )
            )
            hidden = hidden*update_gate + h_hat*(1 - update_gate)
            # add hidden state to output sequence
            out.append(hidden)

        if self.return_hidden:
            return torch.stack(out, dim=0), hidden
        else:
            return torch.stack(out, dim=0)


class ConvGRUCell_bnorm(nn.Module):

    def __init__(self, dims, in_kernel, out_kernel, in_channels, out_channels,
                 learn_initial=False, return_hidden=False):
        super(ConvGRUCell_bnorm, self).__init__()
        self.dims = dims  # input spatial dimentions (H, W)
        self.in_kernel = in_kernel  # 2D input filter kernel shape (H, W)
        self.out_kernel = out_kernel  # 2D hidden filter kernel shape (H, W)
        self.in_channels = in_channels  # number input feature maps
        self.out_channels = out_channels  # number hidden feature maps
        self.in_shape = (out_channels, in_channels, *in_kernel)
        self.out_shape = (out_channels, out_channels, *out_kernel)
        self.in_pad = (in_kernel[0]//2, in_kernel[1]//2)  # conv padding
        self.out_pad = (out_kernel[0]//2, out_kernel[1]//2)  # conv padding
        self.learn_initial = learn_initial  # trainable initial state
        self.return_hidden = return_hidden  # returns (output, hidden)
        self.build()

    def build(self):
        # input weight (transforms X before entering the hidden recurrence)
        self.Wxh = init_filter(self.in_shape)
        self.bxh = nn.Parameter(torch.zeros(self.out_channels).float())
        # hidden weight and bias
        self.Whh = init_filter(self.out_shape)
        self.bhh = nn.Parameter(torch.zeros(self.out_channels).float())
        self.hidden_bnorm = nn.BatchNorm2d(self.out_channels)
        # update gate weights
        self.Wxz = init_filter(self.in_shape)
        self.bxz = nn.Parameter(torch.zeros(self.out_channels).float())
        self.Whz = init_filter(self.out_shape)
        self.bhz = nn.Parameter(torch.zeros(self.out_channels).float())
        self.update_bnorm = nn.BatchNorm2d(self.out_channels)
        # reset gate weights
        self.Wxr = init_filter(self.in_shape)
        self.bxr = nn.Parameter(torch.zeros(self.out_channels).float())
        self.Whr = init_filter(self.out_shape)
        self.bhr = nn.Parameter(torch.zeros(self.out_channels).float())
        self.reset_bnorm = nn.BatchNorm2d(self.out_channels)
        # initial hidden repesentation
        self.h0 = nn.Parameter(
            torch.zeros(self.out_channels, *self.dims).float(),
            requires_grad=self.learn_initial
        )

    def forward(self, X):
        """
        Input shape is (T, batch, C, H, W). Conv2d takes (batch, C, H, W).
        Loop over T and collect output in list, then stack back up on time
        dimension. T is not ragged, videos are all the same duration.
        """
        out = []
        # repeat initial hidden state for each sample
        hidden = self.h0.repeat(X.shape[1], 1, 1, 1)
        for frame in X:
            # calculate gates
            reset_gate = torch.sigmoid(self.reset_bnorm(
                F.conv2d(frame, self.Wxr, bias=self.bxr, padding=self.in_pad)
                + F.conv2d(
                    hidden, self.Whr, bias=self.bhr, padding=self.out_pad
                )
            ))
            update_gate = torch.sigmoid(self.update_bnorm(
                F.conv2d(frame, self.Wxz, bias=self.bxz, padding=self.in_pad)
                + F.conv2d(
                    hidden, self.Whz, bias=self.bhz, padding=self.out_pad
                )
            ))
            # update hidden representation
            h_hat = torch.tanh(self.hidden_bnorm(
                F.conv2d(frame, self.Wxh, bias=self.bxh, padding=self.in_pad)
                + F.conv2d(
                    reset_gate*hidden, self.Whh, bias=self.bhh,
                    padding=self.out_pad
                )
            ))
            hidden = hidden*update_gate + h_hat*(1 - update_gate)
            # add hidden state to output sequence
            out.append(hidden)

        if self.return_hidden:
            return torch.stack(out, dim=0), hidden
        else:
            return torch.stack(out, dim=0)


class ConvGRUCell_bnorm2(nn.Module):

    def __init__(self, dims, in_kernel, out_kernel, in_channels, out_channels,
                 learn_initial=False, return_hidden=False):
        super(ConvGRUCell_bnorm2, self).__init__()
        self.dims = dims  # input spatial dimentions (H, W)
        self.in_kernel = in_kernel  # 2D input filter kernel shape (H, W)
        self.out_kernel = out_kernel  # 2D hidden filter kernel shape (H, W)
        self.in_channels = in_channels  # number input feature maps
        self.out_channels = out_channels  # number hidden feature maps
        self.in_shape = (out_channels, in_channels, *in_kernel)
        self.out_shape = (out_channels, out_channels, *out_kernel)
        self.in_pad = (in_kernel[0]//2, in_kernel[1]//2)  # conv padding
        self.out_pad = (out_kernel[0]//2, out_kernel[1]//2)  # conv padding
        self.learn_initial = learn_initial  # trainable initial state
        self.return_hidden = return_hidden  # returns (output, hidden)
        self.build()

    def build(self):
        # initial gama for batchnorm
        gamma = torch.Tensor([.1]*self.out_channels)
        # input weight (transforms X before entering the hidden recurrence)
        self.Wxh = init_filter(self.in_shape)
        self.bxh = nn.Parameter(torch.zeros(self.out_channels).float())
        self.xh_bnorm = nn.BatchNorm2d(self.out_channels)
        self.xh_bnorm.weight.data = gamma  # initialize
        # hidden weight and bias
        self.Whh = init_filter(self.out_shape)
        self.bhh = nn.Parameter(torch.zeros(self.out_channels).float())
        self.hh_bnorm = nn.BatchNorm2d(self.out_channels)
        self.hh_bnorm.weight.data = gamma  # initialize
        # update gate weights
        self.Wxz = init_filter(self.in_shape)
        self.bxz = nn.Parameter(torch.zeros(self.out_channels).float())
        self.xz_bnorm = nn.BatchNorm2d(self.out_channels)
        self.xz_bnorm.weight.data = gamma  # initialize
        self.Whz = init_filter(self.out_shape)
        self.bhz = nn.Parameter(torch.zeros(self.out_channels).float())
        self.hz_bnorm = nn.BatchNorm2d(self.out_channels)
        self.hz_bnorm.weight.data = gamma  # initialize
        # reset gate weights
        self.Wxr = init_filter(self.in_shape)
        self.bxr = nn.Parameter(torch.zeros(self.out_channels).float())
        self.xr_bnorm = nn.BatchNorm2d(self.out_channels)
        self.xr_bnorm.weight.data = gamma  # initialize
        self.Whr = init_filter(self.out_shape)
        self.bhr = nn.Parameter(torch.zeros(self.out_channels).float())
        self.hr_bnorm = nn.BatchNorm2d(self.out_channels)
        self.hr_bnorm.weight.data = gamma  # initialize
        self.reset_bnorm = nn.BatchNorm2d(self.out_channels)
        # initial hidden repesentation
        self.h0 = nn.Parameter(
            torch.zeros(self.out_channels, *self.dims).float(),
            requires_grad=self.learn_initial
        )

    def forward(self, X):
        """
        Input shape is (T, batch, C, H, W). Conv2d takes (batch, C, H, W).
        Loop over T and collect output in list, then stack back up on time
        dimension. T is not ragged, videos are all the same duration.
        """
        out = []
        # repeat initial hidden state for each sample
        hidden = self.h0.repeat(X.shape[1], 1, 1, 1)
        for frame in X:
            # calculate gates
            reset_gate = torch.sigmoid(
                self.xr_bnorm(
                    F.conv2d(
                        frame, self.Wxr, bias=self.bxr, padding=self.in_pad
                    )
                )
                + self.hr_bnorm(
                    F.conv2d(
                        hidden, self.Whr, bias=self.bhr, padding=self.out_pad
                    )
                )
            )
            update_gate = torch.sigmoid(
                self.xz_bnorm(
                    F.conv2d(
                        frame, self.Wxz, bias=self.bxz, padding=self.in_pad
                    )
                )
                + self.hz_bnorm(
                    F.conv2d(
                        hidden, self.Whz, bias=self.bhz, padding=self.out_pad
                    )
                )
            )
            # update hidden representation
            h_hat = torch.tanh(
                self.xh_bnorm(
                    F.conv2d(
                        frame, self.Wxh, bias=self.bxh, padding=self.in_pad
                    )
                )
                + self.hh_bnorm(
                    F.conv2d(
                        reset_gate*hidden, self.Whh, bias=self.bhh,
                        padding=self.out_pad
                    )
                )
            )
            hidden = hidden*update_gate + h_hat*(1 - update_gate)
            # add hidden state to output sequence
            out.append(hidden)

        if self.return_hidden:
            return torch.stack(out, dim=0), hidden
        else:
            return torch.stack(out, dim=0)


class ConvGRUCell_wnorm(nn.Module):

    def __init__(self, dims, in_kernel, out_kernel, in_channels, out_channels,
                 learn_initial=False, return_hidden=False):
        super(ConvGRUCell_wnorm, self).__init__()
        self.dims = dims  # input spatial dimentions (H, W)
        self.in_kernel = in_kernel  # 2D input filter kernel shape (H, W)
        self.out_kernel = out_kernel  # 2D hidden filter kernel shape (H, W)
        self.in_channels = in_channels  # number input feature maps
        self.out_channels = out_channels  # number hidden feature maps
        self.in_shape = (out_channels, in_channels, *in_kernel)
        self.out_shape = (out_channels, out_channels, *out_kernel)
        self.in_pad = (in_kernel[0]//2, in_kernel[1]//2)  # conv padding
        self.out_pad = (out_kernel[0]//2, out_kernel[1]//2)  # conv padding
        self.learn_initial = learn_initial  # trainable initial state
        self.return_hidden = return_hidden  # returns (output, hidden)
        self.build()

    def build(self):
        # input weight (transforms X before entering the hidden recurrence)
        self.Wxh = init_filter(self.in_shape)
        self.bxh = nn.Parameter(torch.zeros(self.out_channels).float())
        # hidden weight and bias
        self.Whh = init_filter(self.out_shape)
        self.bhh = nn.Parameter(torch.zeros(self.out_channels).float())
        # update gate weights
        self.Wxz = init_filter(self.in_shape)
        self.bxz = nn.Parameter(torch.zeros(self.out_channels).float())
        self.Whz = init_filter(self.out_shape)
        self.bhz = nn.Parameter(torch.zeros(self.out_channels).float())
        # reset gate weights
        self.Wxr = init_filter(self.in_shape)
        self.bxr = nn.Parameter(torch.zeros(self.out_channels).float())
        self.Whr = init_filter(self.out_shape)
        self.bhr = nn.Parameter(torch.zeros(self.out_channels).float())
        # initial hidden repesentation
        self.h0 = nn.Parameter(
            torch.zeros(self.out_channels, *self.dims).float(),
            requires_grad=self.learn_initial
        )
        # setup weight normalization hooks
        self.weight_names = ['Wxh', 'Whh', 'Wxz', 'Whz', 'Wxr', 'Whr']
        for name in self.weight_names:
            self = weight_norm(self, name)

    def forward(self, X):
        """
        Input shape is (T, batch, C, H, W). Conv2d takes (batch, C, H, W).
        Loop over T and collect output in list, then stack back up on time
        dimension. T is not ragged, videos are all the same duration.
        """
        out = []
        # repeat initial hidden state for each sample
        hidden = self.h0.repeat(X.shape[1], 1, 1, 1)
        for frame in X:
            # calculate gates
            reset_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxr, bias=self.bxr, padding=self.in_pad)
                + F.conv2d(
                    hidden, self.Whr, bias=self.bhr, padding=self.out_pad
                )
            )
            update_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxz, bias=self.bxz, padding=self.in_pad)
                + F.conv2d(
                    hidden, self.Whz, bias=self.bhz, padding=self.out_pad
                )
            )
            # update hidden representation
            h_hat = torch.tanh(
                F.conv2d(frame, self.Wxh, bias=self.bxh, padding=self.in_pad)
                + F.conv2d(
                    reset_gate*hidden, self.Whh, bias=self.bhh,
                    padding=self.out_pad
                )
            )
            hidden = hidden*update_gate + h_hat*(1 - update_gate)
            # add hidden state to output sequence
            out.append(hidden)

        if self.return_hidden:
            return torch.stack(out, dim=0), hidden
        else:
            return torch.stack(out, dim=0)


class ConvLSTMCell1(nn.Module):
    "Similar to Pytorch LSTM."
    def __init__(self, dims, in_kernel, out_kernel, in_channels, out_channels,
                 learn_initial=False, return_hidden=False):
        super(ConvLSTMCell1, self).__init__()
        self.dims = dims  # input spatial dimentions (H, W)
        self.in_kernel = in_kernel  # 2D input filter kernel shape (H, W)
        self.out_kernel = out_kernel  # 2D hidden filter kernel shape (H, W)
        self.in_channels = in_channels  # number input feature maps
        self.out_channels = out_channels  # number hidden feature maps
        self.in_shape = (out_channels, in_channels, *in_kernel)
        self.out_shape = (out_channels, out_channels, *out_kernel)
        self.in_pad = (in_kernel[0]//2, in_kernel[1]//2)  # conv padding
        self.out_pad = (out_kernel[0]//2, out_kernel[1]//2)  # conv padding
        self.learn_initial = learn_initial  # trainable initial state
        self.return_hidden = return_hidden  # returns (output, hidden)
        self.build()

    def build(self):
        # input gate weights (and bias)
        self.Wxi = init_filter(self.in_shape)
        self.bxi = nn.Parameter(torch.zeros(self.out_channels).float())
        self.Whi = init_filter(self.out_shape)
        self.bhi = nn.Parameter(torch.zeros(self.out_channels).float())
        # forget gate weights (and bias)
        self.Wxf = init_filter(self.in_shape)
        self.bxf = nn.Parameter(torch.zeros(self.out_channels).float())
        self.Whf = init_filter(self.out_shape)
        self.bhf = nn.Parameter(torch.zeros(self.out_channels).float())
        # memory cell weights (and bias)
        self.Wxc = init_filter(self.in_shape)
        self.bxc = nn.Parameter(torch.zeros(self.out_channels))
        self.Whc = init_filter(self.out_shape)
        self.bhc = nn.Parameter(torch.zeros(self.out_channels))
        # output gate weights (and bias)
        self.Wxo = init_filter(self.in_shape)
        self.bxo = nn.Parameter(torch.zeros(self.out_channels).float())
        self.Who = init_filter(self.out_shape)
        self.bho = nn.Parameter(torch.zeros(self.out_channels).float())
        # initial memory cell and hidden repesentation
        self.h0 = nn.Parameter(
            torch.zeros(self.out_channels, *self.dims).float(),
            requires_grad=self.learn_initial
        )
        self.c0 = nn.Parameter(
            torch.zeros(self.out_channels, *self.dims).float(),
            requires_grad=self.learn_initial
        )

    def forward(self, X):
        """
        Input shape is (T, batch, C, H, W). Conv2d takes (batch, C, H, W).
        Loop over T and collect output in list, then stack back up on time
        dimension. T is not ragged, videos are all the same duration.
        """
        out = []
        # repeat initial hidden and cell states for each sample
        hidden = self.h0.repeat(X.shape[1], 1, 1, 1)
        cell = self.c0.repeat(X.shape[1], 1, 1, 1)
        for frame in X:
            # calculate input and forget gates
            in_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxi, bias=self.bxi, padding=self.in_pad)
                + F.conv2d(
                    hidden, self.Whi, bias=self.bhi, padding=self.out_pad
                )
            )
            forget_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxf, bias=self.bxf, padding=self.in_pad)
                + F.conv2d(
                    hidden, self.Whf, bias=self.bhf, padding=self.out_pad
                )
            )
            # calculate new memory cell value using input and forget gates
            c_hat = torch.tanh(
                F.conv2d(frame, self.Wxc, bias=self.bxc, padding=self.in_pad)
                + F.conv2d(
                    hidden, self.Whc, bias=self.bhc, padding=self.out_pad
                )
            )
            cell = forget_gate*cell + in_gate*c_hat
            # calculate output gate
            out_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxo, bias=self.bxo, padding=self.in_pad)
                + F.conv2d(
                    hidden, self.Who, bias=self.bho, padding=self.out_pad
                )
            )
            # update hidden representation
            hidden = out_gate*torch.tanh(cell)
            # add hidden state to output sequence
            out.append(hidden)

        if self.return_hidden:
            return torch.stack(out, dim=0), (hidden, cell)
        else:
            return torch.stack(out, dim=0)


class ConvLSTMCell2(nn.Module):
    "Based on Shi 2016"
    def __init__(self, dims, in_kernel, out_kernel, in_channels, out_channels,
                 learn_initial=False, return_hidden=False):
        super(ConvLSTMCell2, self).__init__()
        self.dims = dims  # input spatial dimentions (H, W)
        self.in_kernel = in_kernel  # 2D input filter kernel shape (H, W)
        self.out_kernel = out_kernel  # 2D hidden filter kernel shape (H, W)
        self.in_channels = in_channels  # number input feature maps
        self.out_channels = out_channels  # number hidden feature maps
        self.in_shape = (out_channels, in_channels, *in_kernel)
        self.out_shape = (out_channels, out_channels, *out_kernel)
        self.in_pad = (in_kernel[0]//2, in_kernel[1]//2)  # conv padding
        self.out_pad = (out_kernel[0]//2, out_kernel[1]//2)  # conv padding
        self.learn_initial = learn_initial  # trainable initial state
        self.return_hidden = return_hidden  # returns (output, hidden)
        self.build()

    def build(self):
        # input gate weights (and bias)
        self.Wxi = init_filter(self.in_shape)
        self.Whi = init_filter(self.out_shape)
        self.Wci = nn.Parameter(
            torch.randn(self.out_channels, *self.dims).float()
            / np.sqrt(2.0 / np.prod(self.dims))
        )
        self.bi = nn.Parameter(torch.zeros(self.out_channels).float())
        # forget gate weights (and bias)
        self.Wxf = init_filter(self.in_shape)
        self.Whf = init_filter(self.out_shape)
        self.Wcf = nn.Parameter(
            torch.randn(self.out_channels, *self.dims).float()
            / np.sqrt(2.0 / np.prod(self.dims))
        )
        self.bf = nn.Parameter(torch.zeros(self.out_channels).float())
        # memory cell weights (and bias)
        self.Wxc = init_filter(self.in_shape)
        self.Whc = init_filter(self.out_shape)
        self.bc = nn.Parameter(torch.zeros(self.out_channels))
        # output gate weights (and bias)
        self.Wxo = init_filter(self.in_shape)
        self.Who = init_filter(self.out_shape)
        self.Wco = nn.Parameter(
            torch.randn(self.out_channels, *self.dims).float()
            / np.sqrt(2.0 / np.prod(self.dims))
        )
        self.bo = nn.Parameter(torch.zeros(self.out_channels).float())
        # initial memory cell and hidden repesentation
        self.h0 = nn.Parameter(
            torch.zeros(self.out_channels, *self.dims).float(),
            requires_grad=self.learn_initial
        )
        self.c0 = nn.Parameter(
            torch.zeros(self.out_channels, *self.dims).float(),
            requires_grad=self.learn_initial
        )

    def forward(self, X):
        """
        Input shape is (T, batch, C, H, W). Conv2d takes (batch, C, H, W).
        Loop over T and collect output in list, then stack back up on time
        dimension. T is not ragged, videos are all the same duration.
        """
        out = []
        # repeat initial hidden and cell states for each sample
        hidden = self.h0.repeat(X.shape[1], 1, 1, 1)
        cell = self.c0.repeat(X.shape[1], 1, 1, 1)
        for frame in X:
            # calculate input and forget gates
            in_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxi, bias=self.bi, padding=self.in_pad)
                + F.conv2d(hidden, self.Whi, padding=self.out_pad)
                + self.Wci*cell
            )
            forget_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxf, bias=self.bf, padding=self.in_pad)
                + F.conv2d(hidden, self.Whf, padding=self.out_pad)
                + self.Wcf*cell
            )
            # calculate new memory cell value using input and forget gates
            c_hat = torch.tanh(
                F.conv2d(frame, self.Wxc, bias=self.bc, padding=self.in_pad)
                + F.conv2d(hidden, self.Whc, padding=self.out_pad)
            )
            cell = forget_gate*cell + in_gate*c_hat
            # calculate output gate
            out_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxo, bias=self.bo, padding=self.in_pad)
                + F.conv2d(hidden, self.Who, padding=self.out_pad)
                + self.Wco*cell
            )
            # update hidden representation
            hidden = out_gate*torch.tanh(cell)
            # add hidden state to output sequence
            out.append(hidden)

        if self.return_hidden:
            return torch.stack(out, dim=0), (hidden, cell)
        else:
            return torch.stack(out, dim=0)


class ConvLSTMCell2_bnorm(nn.Module):
    "Based on Shi 2016, with batchnorm added."
    def __init__(self, dims, in_kernel, out_kernel, in_channels, out_channels,
                 learn_initial=False, return_hidden=False):
        super(ConvLSTMCell2_bnorm, self).__init__()
        self.dims = dims  # input spatial dimentions (H, W)
        self.in_kernel = in_kernel  # 2D input filter kernel shape (H, W)
        self.out_kernel = out_kernel  # 2D hidden filter kernel shape (H, W)
        self.in_channels = in_channels  # number input feature maps
        self.out_channels = out_channels  # number hidden feature maps
        self.in_shape = (out_channels, in_channels, *in_kernel)
        self.out_shape = (out_channels, out_channels, *out_kernel)
        self.in_pad = (in_kernel[0]//2, in_kernel[1]//2)  # conv padding
        self.out_pad = (out_kernel[0]//2, out_kernel[1]//2)  # conv padding
        self.learn_initial = learn_initial  # trainable initial state
        self.return_hidden = return_hidden  # returns (output, hidden)
        self.build()

    def build(self):
        # initial gama for batchnorm
        gamma = torch.Tensor([.1]*self.out_channels)
        # input gate weights (and bias)
        self.Wxi = init_filter(self.in_shape)
        self.xi_bnorm = nn.BatchNorm2d(self.out_channels)
        self.xi_bnorm.weight.data = gamma  # initialize
        self.Whi = init_filter(self.out_shape)
        self.hi_bnorm = nn.BatchNorm2d(self.out_channels)
        self.hi_bnorm.weight.data = gamma  # initialize
        self.Wci = nn.Parameter(
            torch.randn(self.out_channels, *self.dims).float()
            / np.sqrt(2.0 / np.prod(self.dims))
        )
        self.bi = nn.Parameter(torch.zeros(self.out_channels).float())
        # forget gate weights (and bias)
        self.Wxf = init_filter(self.in_shape)
        self.xf_bnorm = nn.BatchNorm2d(self.out_channels)
        self.xf_bnorm.weight.data = gamma  # initialize
        self.Whf = init_filter(self.out_shape)
        self.hf_bnorm = nn.BatchNorm2d(self.out_channels)
        self.hf_bnorm.weight.data = gamma  # initialize
        self.Wcf = nn.Parameter(
            torch.randn(self.out_channels, *self.dims).float()
            / np.sqrt(2.0 / np.prod(self.dims))
        )
        self.bf = nn.Parameter(torch.zeros(self.out_channels).float())
        # memory cell weights (and bias)
        self.Wxc = init_filter(self.in_shape)
        self.xc_bnorm = nn.BatchNorm2d(self.out_channels)
        self.xc_bnorm.weight.data = gamma  # initialize
        self.Whc = init_filter(self.out_shape)
        self.hc_bnorm = nn.BatchNorm2d(self.out_channels)
        self.hc_bnorm.weight.data = gamma  # initialize
        self.bc = nn.Parameter(torch.zeros(self.out_channels))
        # output gate weights (and bias)
        self.Wxo = init_filter(self.in_shape)
        self.xo_bnorm = nn.BatchNorm2d(self.out_channels)
        self.xo_bnorm.weight.data = gamma  # initialize
        self.Who = init_filter(self.out_shape)
        self.ho_bnorm = nn.BatchNorm2d(self.out_channels)
        self.ho_bnorm.weight.data = gamma  # initialize
        self.Wco = nn.Parameter(
            torch.randn(self.out_channels, *self.dims).float()
            / np.sqrt(2.0 / np.prod(self.dims))
        )
        self.bo = nn.Parameter(torch.zeros(self.out_channels).float())
        # initial memory cell and hidden repesentation
        self.h0 = nn.Parameter(
            torch.zeros(self.out_channels, *self.dims).float(),
            requires_grad=self.learn_initial
        )
        self.c0 = nn.Parameter(
            torch.zeros(self.out_channels, *self.dims).float(),
            requires_grad=self.learn_initial
        )

    def forward(self, X):
        """
        Input shape is (T, batch, C, H, W). Conv2d takes (batch, C, H, W).
        Loop over T and collect output in list, then stack back up on time
        dimension. T is not ragged, videos are all the same duration.
        """
        out = []
        # repeat initial hidden and cell states for each sample
        hidden = self.h0.repeat(X.shape[1], 1, 1, 1)
        cell = self.c0.repeat(X.shape[1], 1, 1, 1)
        for frame in X:
            # calculate input and forget gates
            in_gate = torch.sigmoid(
                self.xi_bnorm(
                    F.conv2d(
                        frame, self.Wxi, bias=self.bi, padding=self.in_pad
                    )
                )
                + self.hi_bnorm(
                    F.conv2d(hidden, self.Whi, padding=self.out_pad)
                )
                + self.Wci*cell
            )
            forget_gate = torch.sigmoid(
                self.xf_bnorm(
                    F.conv2d(
                        frame, self.Wxf, bias=self.bf, padding=self.in_pad
                    )
                )
                + self.hf_bnorm(
                    F.conv2d(hidden, self.Whf, padding=self.out_pad)
                )
                + self.Wcf*cell
            )
            # calculate new memory cell value using input and forget gates
            c_hat = torch.tanh(
                self.xc_bnorm(
                    F.conv2d(
                        frame, self.Wxc, bias=self.bc, padding=self.in_pad
                    )
                )
                + self.hc_bnorm(
                    F.conv2d(hidden, self.Whc, padding=self.out_pad)
                )
            )
            cell = forget_gate*cell + in_gate*c_hat
            # calculate output gate
            out_gate = torch.sigmoid(
                self.xo_bnorm(
                    F.conv2d(
                        frame, self.Wxo, bias=self.bo, padding=self.in_pad
                    )
                )
                + self.ho_bnorm(
                    F.conv2d(hidden, self.Who, padding=self.out_pad)
                )
                + self.Wco*cell
            )
            # update hidden representation
            hidden = out_gate*torch.tanh(cell)
            # add hidden state to output sequence
            out.append(hidden)

        if self.return_hidden:
            return torch.stack(out, dim=0), (hidden, cell)
        else:
            return torch.stack(out, dim=0)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # generate random data and build Convolutional RNN cells
    data = torch.randn(60, 5, 10, 30, 30)  # (T, N, C, H, W)
    gru = ConvGRUCell_wnorm(
        (30, 30),  # input dimensions  (H, W)
        (3, 3),  # input weight filter kernel
        (3, 3),  # output weight filter kernel
        10,  # input channels
        20  # hidden/output channels
    ).to(device)
    lstm = ConvLSTMCell2((30, 30), (3, 3), (3, 3), 10, 20).to(device)

    # run data through Convolutional RNNs
    gru_out, gru_h = gru(data.to(device))
    lstm_out, (lstm_h, lstm_c) = lstm(data.to(device))

    # convert data and outputs into numpy
    data = data.detach().numpy()
    gru_out = gru_out.detach().cpu().numpy()
    gru_h = gru_h.detach().cpu().numpy()
    lstm_out = lstm_out.detach().cpu().numpy()
    lstm_h = lstm_h.detach().cpu().numpy()
    lstm_c = lstm_c.detach().cpu().numpy()

    # input
    print('input dimensions (T, N, C, H, W)')
    print('input shape:', data.shape, end='\n\n')
    # sequence outputs
    print('output dimensions (T, N, C, H, W)')
    print('gru output shape:', gru_out.shape)
    print('lstm output shape:', lstm_out.shape, end='\n\n')
    # final states
    print('state dimensions (N, C, H, W)')
    print('gru hidden shape:', gru_h.shape)
    print('lstm hidden shape:', lstm_h.shape)
    print('lstm cell shape:', lstm_c.shape)
