# import numpy as np
import matplotlib.pyplot as plt

import recurrent_convolution as crnns
from temporal_convolution import TemporalConv3dStack
from sim_util import StackPlotter

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from custom_loss import DecoderLoss
from retina_dataset import RetinaVideos
from torch.utils.data import DataLoader

"""
Thoughts:
- try regular convolutions after (or interleaved) with transpose convolutions
    to smooth out the blockiness that arises from the upsampling.
- try using lists of dicts for layer params to try and
    make the model's build() read more clearly.
"""


class RetinaDecoder(nn.Module):

    def __init__(self, grp_tempo_params, conv_params, crnn_cell_params,
                 temp3d_stack_params,
                 trans_params, post_conv_params, crnn_cell=crnns.ConvGRUCell,
                 learn_initial=False):
        super(RetinaDecoder, self).__init__()
        # layer parameters
        self.grp_tempo_params = grp_tempo_params
        self.conv_params = conv_params
        self.crnn_cell_params = crnn_cell_params
        self.temp3d_stack_params = temp3d_stack_params
        self.trans_params = trans_params
        self.post_conv_params = post_conv_params
        # ConvRNN settings
        self.crnn_cell = crnn_cell
        self.learn_initial = learn_initial
        # create model and send to GPU
        self.build()
        self.to(device)

    def build(self):
        # grouped convolutions
        self.grp_tempo_layers = nn.ModuleList()
        self.grp_tempo_bnorms = nn.ModuleList()
        # convolutions
        self.conv_layers = nn.ModuleList()
        self.conv_bnorms = nn.ModuleList()
        # recurrent convolutions
        self.crnn_stack = nn.ModuleList()
        # 3d temporal convolutions
        self.tempo3d_layers = nn.ModuleList()
        # transpose convolutions
        self.trans_layers = nn.ModuleList()
        self.trans_bnorms = nn.ModuleList()
        # post-upsampling convolutions
        self.post_conv_layers = nn.ModuleList()
        self.post_conv_bnorms = nn.ModuleList()

        for params in self.grp_tempo_params:
            # [in, [block_channel(s)], (D, H, W), stride, dilation, groups]
            self.grp_tempo_layers.append(
                TemporalConv3dStack(*params[:-1], activation=params[-1])
            )
            # self.grp_conv_bnorms.append(nn.BatchNorm3d(params[1]))

        for params in self.conv_params:
            # params: [in, out, (kernel), (stride)]
            pad = (params[2][0]//2, params[2][1]//2, params[2][2]//2)
            self.conv_layers.append(nn.Conv3d(*params, pad))
            self.conv_bnorms.append(nn.BatchNorm3d(params[1]))

        for params in self.crnn_cell_params:
            # params: [(dims), (in_kernel), (out_kernel), in_C, out_C]
            # recurrenct convolutional cells (GRU or LSTM)
            self.crnn_stack.append(
                self.crnn_cell(
                    *params, learn_initial=self.learn_initial
                )
            )

        for params in self.temp3d_stack_params:
            self.tempo3d_layers.append(
                TemporalConv3dStack(*params[:3], activation=params[3])
            )

        for params in self.trans_params:
            pad = (params[2][0]//2, params[2][1]//2, params[2][2]//2)
            self.trans_layers.append(
                nn.ConvTranspose3d(*params, padding=pad, output_padding=pad)
            )
            self.trans_bnorms.append(nn.BatchNorm3d(params[1]))

        for params in self.post_conv_params:
            # params: [in, out, (kernel), (stride)]
            pad = (params[2][0]//2, params[2][1]//2, params[2][2]//2)
            self.post_conv_layers.append(nn.Conv3d(*params, pad))
            self.post_conv_bnorms.append(nn.BatchNorm3d(params[1]))

    def forward(self, X):
        # time to 'depth' dimension
        X = X.permute(1, 2, 0, 3, 4)  # to (N, C, T, H, W)

        # reduce spatial dimensionality (collate somatic information)
        X = F.avg_pool3d(X, (1, 2, 2))

        # grouped (cluster siloed) temporal convolutions
        for tempo_conv in self.grp_tempo_layers:
            X = tempo_conv(X)

        # frame-by-frame (space only) convolutions
        for conv, bnorm in zip(self.conv_layers, self.conv_bnorms):
            X = torch.tanh(bnorm(conv(X)))
        # X = F.avg_pool3d(X, (1, 2, 2))
        # testing! (try max if using ReLU at the start)
        X = F.max_pool3d(X, (1, 2, 2))

        if len(self.crnn_stack) > 0:
            # return to time dimension first for operations over time
            X = X.permute(2, 0, 1, 3, 4)  # back to (T, N, C, H, w)

            # stacked convolutional recurrent cells
            for cell in self.crnn_stack:
                X, _ = cell(X)

            # expand back out in space and reduce channels
            X = X.permute(1, 2, 0, 3, 4)  # time to 'depth' dimension

        for tempo_conv in self.tempo3d_layers:
            X = tempo_conv(X)

        for trans, bnorm in zip(self.trans_layers, self.trans_bnorms):
            X = torch.tanh(bnorm(trans(X)))

        # clean up with more spatial convs (try interleaving with trans next)
        # frame-by-frame (space only) convolutions
        for conv, bnorm in zip(self.post_conv_layers, self.post_conv_bnorms):
            X = torch.tanh(bnorm(conv(X)))

        X = X.permute(2, 0, 1, 3, 4)  # back to (T, N, C, H, w)

        return X

    def fit(self, train_set, test_set, lr=1e-4, epochs=10, batch_sz=1,
            print_every=40):

        train_loader = DataLoader(
            train_set, batch_size=batch_sz, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_sz, num_workers=2
        )
        N = train_set.__len__()  # number of samples

        # DecoderLoss equivalent to MSE when alpha=0
        self.loss = DecoderLoss(alpha=10).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        n_batches = N // batch_sz
        train_costs, test_costs = [], []
        for i in range(epochs):
            cost = 0
            print("epoch:", i, "n_batches:", n_batches)
            for j, batch in enumerate(train_loader):
                net, stim = batch['net'].to(device), batch['stim'].to(device)
                cost += self.train_step(net.transpose(0, 1), stim)
                del net, stim, batch

                if j % print_every == 0:
                    # costs and accuracies for test set
                    test_cost = 0
                    for t, testB in enumerate(test_loader, 1):
                        net = testB['net'].to(device)
                        stim = testB['stim'].to(device)
                        testB_cost = self.get_cost(net.transpose(0, 1), stim)
                        del net, stim, testB
                        test_cost += testB_cost
                    test_cost /= t+1

                    print("cost: %f" % (test_cost))

            # for plotting
            train_costs.append(cost / n_batches)
            test_costs.append(test_cost)

        # plot cost and accuracy progression
        fig, axes = plt.subplots(1)
        axes.plot(train_costs, label='training')
        axes.plot(test_costs, label='validation')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Cost')
        plt.legend()
        plt.show()

    def train_step(self, inputs, targets):
        self.train()  # set the model to training mode
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        decoded = self.forward(inputs)
        output = self.loss.forward(
            # swap batch to first dimension
            decoded.transpose(0, 1), targets
        )

        # Backward
        output.backward()  # compute gradients
        self.optimizer.step()  # Update parameters

        return output.item()  # cost

    def get_cost(self, inputs, targets):
        self.eval()  # set the model to testing mode
        self.optimizer.zero_grad()  # Reset gradient
        with torch.no_grad():
            # Forward
            decoded = self.forward(inputs)
            output = self.loss.forward(
                # swap batch to first dimension
                decoded.transpose(0, 1), targets
            )
        return output.item()

    def decode(self, sample_set):
        self.eval()  # set the model to testing mode
        sample_loader = DataLoader(
            sample_set, batch_size=1, shuffle=True, num_workers=2
        )
        for i, sample in enumerate(sample_loader):
            with torch.no_grad():
                # get stimulus prediction from network activity
                decoded = self.forward(sample['net'].to(device))

            # Reduce out batch and channel dims, then put time last
            # (T, N, C, H, W) -> (H, W, T)
            decoded = decoded.squeeze().cpu().numpy().transpose(1, 2, 0)
            net = sample['net'].squeeze().numpy().sum(axis=1)
            net = net.transpose(1, 2, 0)
            stim = sample['stim'].squeeze().numpy().transpose(1, 2, 0)

            # synced scrollable videos of cell actity, decoding, and stimulus
            fig, ax = plt.subplots(1, 3, figsize=(17, 6))
            net_stack = StackPlotter(ax[0], net, delta=1, vmin=0)
            deco_stack = StackPlotter(ax[1], decoded, delta=1, vmin=-1, vmax=1)
            stim_stack = StackPlotter(ax[2], stim, delta=1, vmin=-1, vmax=1)
            fig.canvas.mpl_connect('scroll_event', net_stack.onscroll)
            fig.canvas.mpl_connect('scroll_event', deco_stack.onscroll)
            fig.canvas.mpl_connect('scroll_event', stim_stack.onscroll)
            ax[0].set_title('Network Recording')
            ax[1].set_title('Decoding')
            ax[2].set_title('Stimulus')
            fig.tight_layout()
            plt.show()

            again = input(
                "Show another reconstruction? Enter 'n' to quit\n")
            if again == 'n':
                break


def decoder_setup_1():
    decoder = RetinaDecoder(
        # grouped temporal conv stacks:
        # [in, [block_channel(s)], (D, H, W), stride, dilation, groups]
        [

        ],
        # spatial conv layers: [in, out, (D, H, W), stride]
        [
            [15, 64, (1, 3, 3), (1, 1, 1)],
            [64, 128, (1, 3, 3), (1, 1, 1)],
            [128, 64, (1, 3, 3), (1, 1, 1)],
        ],
        # for each ConvRNN cell:
        # [spatial_dims, in_kernel, out_kernel, in_channels, out_channels]
        [

        ],
        # temporal convolution stack(s)
        [
            [64, [128, 256, 128], (2, 3, 3), nn.Tanh],
        ],
        # ConvTranspose layers: [in, out, (D, H, W), stride]
        [
            [128, 64, (1, 3, 3), (1, 2, 2)],
            [64, 1, (1, 3, 3), (1, 2, 2)],
        ],
        # post conv layers
        [

        ],
        crnn_cell=crnns.ConvGRUCell_bnorm2,
        learn_initial=False
    )
    return decoder


def decoder_setup_2():
    decoder = RetinaDecoder(
        # grouped temporal conv stacks:
        # [in, [block_channel(s)], (D, H, W), stride, dilation, groups]
        [
            # [15, [45, 45, 15], (2, 1, 1), 1, 15, nn.Tanh]
            [15, [45, 45, 15], (2, 1, 1), 1, 15, nn.ReLU]
        ],
        # spatial conv layers: [in, out, (D, H, W), stride]
        [
            # [15, 64, (1, 3, 3), (1, 1, 1)],
        ],
        # for each ConvRNN cell:
        # [spatial_dims, in_kernel, out_kernel, in_channels, out_channels]
        [

        ],
        # temporal convolution stack(s)
        [
            # [15, [128, 256, 128], (2, 3, 3), nn.Tanh],
            [15, [128, 256, 128], (2, 3, 3), nn.ReLU],
        ],
        # ConvTranspose layers: [in, out, (D, H, W), stride]
        [
            [128, 64, (1, 3, 3), (1, 2, 2)],
            [64, 1, (1, 3, 3), (1, 2, 2)],
        ],
        # post conv layers
        [

        ],
        crnn_cell=crnns.ConvGRUCell_bnorm2,
        learn_initial=False
    )
    return decoder


def decoder_setup_3():
    decoder = RetinaDecoder(
        # grouped temporal conv stacks:
        # [in, [block_channel(s)], (D, H, W), stride, dilation, groups]
        [
            [15, [45, 45, 15], (2, 1, 1), 1, 15, nn.ReLU]
        ],
        # spatial conv layers: [in, out, (D, H, W), stride]
        [
            # [15, 128, (1, 3, 3), (1, 1, 1)],
        ],
        # for each ConvRNN cell:
        # [spatial_dims, in_kernel, out_kernel, in_channels, out_channels]
        [
            [(25, 25), (3, 3), (3, 3), 15, 64],
        ],
        # temporal convolution stack(s)
        [

        ],
        # ConvTranspose layers: [in, out, (D, H, W), stride]
        [
            [64, 64, (1, 3, 3), (1, 2, 2)],
            [64, 1, (1, 3, 3), (1, 2, 2)],
        ],
        # post conv layers
        [

        ],
        # attempted weight_norm use seems to be causing NaN loss
        # read up about using in custom modules, might be missing something
        crnn_cell=crnns.ConvGRUCell_wnorm,
        learn_initial=False
    )
    return decoder


def main():
    train_path = 'D:/retina-sim-data/second/train_video_dataset/'
    test_path = 'D:/retina-sim-data/second/test_video_dataset/'

    print('Building datasets...')
    train_set = RetinaVideos(train_path, preload=False, crop_centre=[100, 100])
    test_set = RetinaVideos(test_path, preload=True, crop_centre=[100, 100])

    print('Building model...')
    decoder = decoder_setup_2()

    print('Fitting model...')
    decoder.fit(
        train_set, test_set, lr=1e-2, epochs=5, batch_sz=4, print_every=80
    )
    print('Training set examples...')
    decoder.decode(train_set)
    print('Validation set examples...')
    decoder.decode(test_set)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    main()
