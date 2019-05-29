import torch
from torch import nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):

    def __init__(self, dims, in_kernel, out_kernel, in_channels, out_channels):
        super(ConvGRUCell, self).__init__()
        self.dims = dims  # input spatial dimentions (H, W)
        self.in_kernel = in_kernel  # 2D input filter kernel shape (H, W)
        self.out_kernel = out_kernel  # 2D hidden filter kernel shape (H, W)
        self.in_channels = in_channels  # number input feature maps
        self.out_channels = out_channels  # number hidden feature maps
        self.in_shape = (out_channels, in_channels, *in_kernel)
        self.out_shape = (out_channels, out_channels, *out_kernel)
        self.in_pad = (in_kernel[0]//2, in_kernel[1]//2)
        self.out_pad = (out_kernel[0]//2, out_kernel[1]//2)
        self.build()

    def build(self):
        # input weight (transforms X before entering the hidden recurrence)
        self.Wxh = nn.Parameter(torch.randn(*self.in_shape).float())
        # hidden weight and bias
        self.Whh = nn.Parameter(torch.randn(*self.out_shape).float())
        self.bh = nn.Parameter(torch.zeros(self.out_channels).float())
        # update gate weights
        self.Wxz = nn.Parameter(torch.randn(*self.in_shape).float())
        self.Whz = nn.Parameter(torch.randn(*self.out_shape).float())
        self.bz = nn.Parameter(torch.zeros(self.out_channels).float())
        # reset gate weights
        self.Wxr = nn.Parameter(torch.randn(*self.in_shape).float())
        self.Whr = nn.Parameter(torch.randn(*self.out_shape).float())
        self.br = nn.Parameter(torch.zeros(self.out_channels).float())
        # initial hidden repesentation
        self.h0 = nn.Parameter(
            torch.randn(self.out_channels, *self.dims).float()
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
            reset = torch.sigmoid(
                F.conv2d(frame, self.Wxr, bias=self.br, padding=self.in_pad)
                + F.conv2d(hidden, self.Whr, padding=self.out_pad)
            )
            update = torch.sigmoid(
                F.conv2d(frame, self.Wxz, bias=self.bz, padding=self.in_pad)
                + F.conv2d(hidden, self.Whz, padding=self.out_pad)
            )
            # update hidden representation
            h_hat = F.relu(
                F.conv2d(frame, self.Wxh, bias=self.bh, padding=self.in_pad)
                + F.conv2d(reset*hidden, self.Whh, padding=self.out_pad)
            )
            hidden = h_hat*update + hidden*(1-update)
            # add hidden state to output sequence
            out.append(hidden)

        return torch.stack(out, dim=0), hidden


class ConvLSTMCell(nn.Module):

    def __init__(self, dims, in_kernel, out_kernel, in_channels, out_channels):
        super(ConvLSTMCell, self).__init__()
        self.dims = dims  # input spatial dimentions (H, W)
        self.in_kernel = in_kernel  # 2D input filter kernel shape (H, W)
        self.out_kernel = out_kernel  # 2D hidden filter kernel shape (H, W)
        self.in_channels = in_channels  # number input feature maps
        self.out_channels = out_channels  # number hidden feature maps
        self.in_shape = (out_channels, in_channels, *in_kernel)
        self.out_shape = (out_channels, out_channels, *out_kernel)
        self.in_pad = (in_kernel[0]//2, in_kernel[1]//2)
        self.out_pad = (out_kernel[0]//2, out_kernel[1]//2)
        self.build()

    def build(self):
        # input gate weights (and bias)
        self.Wxi = nn.Parameter(torch.randn(*self.in_shape).float())
        self.Whi = nn.Parameter(torch.randn(*self.out_shape).float())
        self.Wci = nn.Parameter(torch.randn(*self.out_shape).float())
        self.bi = nn.Parameter(torch.zeros(self.out_channels).float())
        # forget gate weights (and bias)
        self.Wxf = nn.Parameter(torch.randn(*self.in_shape).float())
        self.Whf = nn.Parameter(torch.randn(*self.out_shape).float())
        self.Wcf = nn.Parameter(torch.randn(*self.out_shape).float())
        self.bf = nn.Parameter(torch.zeros(self.out_channels).float())
        # memory cell weights (and bias)
        self.Wxc = nn.Parameter(torch.randn(*self.in_shape).float())
        self.Whc = nn.Parameter(torch.randn(*self.out_shape).float())
        self.bc = nn.Parameter(torch.zeros(self.out_channels))
        # output gate weights (and bias)
        self.Wxo = nn.Parameter(torch.randn(*self.in_shape).float())
        self.Who = nn.Parameter(torch.randn(*self.out_shape).float())
        self.Wco = nn.Parameter(torch.randn(*self.out_shape).float())
        self.bo = nn.Parameter(torch.zeros(self.out_channels).float())
        # initial memory cell and hidden repesentation
        self.h0 = nn.Parameter(
            torch.randn(self.out_channels, *self.dims).float()
        )
        self.c0 = nn.Parameter(
            torch.randn(self.out_channels, *self.dims).float()
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
