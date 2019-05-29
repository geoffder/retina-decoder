import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


def init_filter(shape):
    "Initialize a conv2d filter. Shape (out_channels, in_channels, H, W)."
    W = torch.randn(*shape).float() / np.sqrt(2.0 / np.prod(shape[:-1]))
    return nn.Parameter(W)


class ConvGRUCell(nn.Module):

    def __init__(self, dims, in_kernel, out_kernel, in_channels, out_channels,
                 learn_initial=True):
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
        self.build()

    def build(self):
        # input weight (transforms X before entering the hidden recurrence)
        self.Wxh = init_filter(self.in_shape)
        # hidden weight and bias
        self.Whh = init_filter(self.out_shape)
        self.bh = nn.Parameter(torch.zeros(self.out_channels).float())
        # update gate weights
        self.Wxz = init_filter(self.in_shape)
        self.Whz = init_filter(self.out_shape)
        self.bz = nn.Parameter(torch.zeros(self.out_channels).float())
        # reset gate weights
        self.Wxr = init_filter(self.in_shape)
        self.Whr = init_filter(self.out_shape)
        self.br = nn.Parameter(torch.zeros(self.out_channels).float())
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
                F.conv2d(frame, self.Wxr, bias=self.br, padding=self.in_pad)
                + F.conv2d(hidden, self.Whr, padding=self.out_pad)
            )
            update_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxz, bias=self.bz, padding=self.in_pad)
                + F.conv2d(hidden, self.Whz, padding=self.out_pad)
            )
            # update hidden representation
            h_hat = F.relu(
                F.conv2d(frame, self.Wxh, bias=self.bh, padding=self.in_pad)
                + F.conv2d(reset_gate*hidden, self.Whh, padding=self.out_pad)
            )
            hidden = h_hat*update_gate + hidden*(1 - update_gate)
            # add hidden state to output sequence
            out.append(hidden)

        return torch.stack(out, dim=0), hidden


class ConvLSTMCell(nn.Module):

    def __init__(self, dims, in_kernel, out_kernel, in_channels, out_channels,
                 learn_initial=True):
        super(ConvLSTMCell, self).__init__()
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
        self.build()

    def build(self):
        # input gate weights (and bias)
        self.Wxi = init_filter(self.in_shape)
        self.Whi = init_filter(self.out_shape)
        self.Wci = init_filter(self.out_shape)
        self.bi = nn.Parameter(torch.zeros(self.out_channels).float())
        # forget gate weights (and bias)
        self.Wxf = init_filter(self.in_shape)
        self.Whf = init_filter(self.out_shape)
        self.Wcf = init_filter(self.out_shape)
        self.bf = nn.Parameter(torch.zeros(self.out_channels).float())
        # memory cell weights (and bias)
        self.Wxc = init_filter(self.in_shape)
        self.Whc = init_filter(self.out_shape)
        self.bc = nn.Parameter(torch.zeros(self.out_channels))
        # output gate weights (and bias)
        self.Wxo = init_filter(self.in_shape)
        self.Who = init_filter(self.out_shape)
        self.Wco = init_filter(self.out_shape)
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
                + F.conv2d(cell, self.Wci, padding=self.out_pad)
            )
            forget_gate = torch.sigmoid(
                F.conv2d(frame, self.Wxf, bias=self.bi, padding=self.in_pad)
                + F.conv2d(hidden, self.Whf, padding=self.out_pad)
                + F.conv2d(cell, self.Wcf, padding=self.out_pad)
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
                + F.conv2d(cell, self.Wco, padding=self.out_pad)
            )
            # update hidden representation
            hidden = out_gate*torch.tanh(cell)
            # add hidden state to output sequence
            out.append(hidden)

        return torch.stack(out, dim=0), (hidden, cell)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # generate random data and build Convolutional RNN cells
    data = torch.randn(60, 5, 10, 30, 30)
    gru = ConvGRUCell((30, 30), (3, 3), (3, 3), 10, 20).to(device)
    lstm = ConvLSTMCell((30, 30), (3, 3), (3, 3), 10, 20).to(device)

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
