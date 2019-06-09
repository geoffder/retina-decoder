import numpy as np
import matplotlib.pyplot as plt

import os

import recurrent_convolution as crnns
from temporal_convolution import TemporalConv3dStack, CausalTranspose3d
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
- trying out temporal downsampling paired with upsampling right now. Make sure
    to explore this before discounting it.
- think about adding conditionals after each encoder block that will setup
    pool functions/layers if specified with a param dict {}. Would make it more
    flexible.
"""


class RetinaDecoder(nn.Module):

    def __init__(self, grp_tempo_params, conv_params, crnn_cell_params,
                 temp3d_stack_params, trans_params, post_conv_params):
        super(RetinaDecoder, self).__init__()
        # layer parameters
        self.grp_tempo_params = grp_tempo_params
        self.conv_params = conv_params
        self.crnn_cell_params = crnn_cell_params
        self.temp3d_stack_params = temp3d_stack_params
        self.trans_params = trans_params
        self.post_conv_params = post_conv_params
        # create model and send to GPU
        self.build()
        self.to(device)

    def build(self):
        # grouped convolutions
        self.grp_tempo_layers = nn.ModuleList()
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

        for p in self.grp_tempo_params:
            self.grp_tempo_layers.append(
                TemporalConv3dStack(
                    p['in'], p['out'], p.get('kernel', (2, 1, 1)),
                    p.get('space_dilation', 1), p.get('groups', 1),
                    p.get('dropout', 0), p.get('activation', nn.ReLU)
                )
            )

        for p in self.conv_params:
            d, h, w = p.get('kernel', (1, 3, 3))
            pad = (d//2, h//2, w//2)
            self.conv_layers.append(
                nn.Conv3d(
                    p['in'], p['out'], (d, h, w), p.get('stride', 1), pad,
                    p.get('dilation', 1), p.get('groups', 1),
                    p.get('bias', True)
                )
            )
            self.conv_bnorms.append(nn.BatchNorm3d(p['out']))

        for p in self.crnn_cell_params:
            # recurrenct convolutional cells (GRU or LSTM)
            self.crnn_stack.append(
                p.get('crnn_cell', crnns.ConvGRUCell_wnorm)(
                    p['dims'], p['in_kernel'], p['out_kernel'], p['in'],
                    p['out'], p.get('learn_initial', False)
                )
            )

        for p in self.temp3d_stack_params:
            self.tempo3d_layers.append(
                TemporalConv3dStack(
                    p['in'], p['out'], p.get('kernel', (2, 3, 3)),
                    p.get('space_dilation', 1), p.get('groups', 1),
                    p.get('dropout', 0), p.get('activation', nn.ReLU)
                )
            )

        for p in self.trans_params:
            self.trans_layers.append(
                CausalTranspose3d(
                    p['in'], p['out'], p['kernel'], p['stride'],
                    p.get('groups', 1), p.get('bias', True),
                    p.get('dilations', (1, 1, 1))
                )
            )
            self.trans_bnorms.append(nn.BatchNorm3d(p['out']))

        for p in self.post_conv_params:
            d, h, w = p.get('kernel', (1, 3, 3))
            pad = (d//2, h//2, w//2)
            self.post_conv_layers.append(
                nn.Conv3d(
                    p['in'], p['out'], (d, h, w), p.get('stride', 1), pad,
                    p.get('dilation', 1), p.get('groups', 1),
                    p.get('bias', True)
                )
            )
            self.post_conv_bnorms.append(nn.BatchNorm3d(p['out']))

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
        X = F.avg_pool3d(X, (2, 2, 2))
        # testing! (try max if using ReLU at the start)
        # X = F.max_pool3d(X, (1, 2, 2))

        if len(self.crnn_stack) > 0:
            # return to time dimension first for operations over time
            X = X.permute(2, 0, 1, 3, 4)  # back to (T, N, C, H, w)

            # stacked convolutional recurrent cells
            for cell in self.crnn_stack:
                X = cell(X)

            X = F.relu(X)  # test

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
            loss_alpha=10, print_every=40):

        train_loader = DataLoader(
            train_set, batch_size=batch_sz, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_sz, num_workers=2
        )
        N = train_set.__len__()  # number of samples

        # DecoderLoss equivalent to MSE when alpha=0 (original default: 10)
        self.loss = DecoderLoss(alpha=loss_alpha).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-8)

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
                net = sample['net'].to(device).transpose(0, 1)
                decoded = self.forward(net)
                del net

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

    def save_decodings(self, sample_set):
        self.eval()  # set the model to testing mode
        sample_loader = DataLoader(sample_set, batch_size=1, num_workers=2)

        while True:
            nametag = input("Decoding set name: ")
            basefold = os.path.join(sample_set.root_dir, nametag)
            if not os.path.isdir(basefold):
                os.mkdir(basefold)
                break
            else:
                print('Folder exists, provide another name...')

        for i, sample in enumerate(sample_loader):
            with torch.no_grad():
                # get stimulus prediction from network activity
                net = sample['net'].to(device).transpose(0, 1)
                decoded = self.forward(net)
                del sample, net

            # Reduce out batch and channel dims
            # (T, N, C, H, W) -> (T, H, W)
            decoded = decoded.squeeze().cpu().numpy()

            # save into subfolder corresponding to originating network
            decofold = os.path.join(
                basefold, sample_set.rec_frame.iloc[i, 0],  # net folder name
            )
            if not os.path.isdir(decofold):
                os.mkdir(decofold)
            np.save(
                # file name corresponding to stimulus
                os.path.join(decofold, sample_set.rec_frame.iloc[i, 1]),
                decoded
            )


def decoder_setup_1():
    "Playing with spatial convs after transpose convolutions."
    decoder = RetinaDecoder(
        # grouped temporal conv stacks:
        [
            {
                'in': 15, 'out': [45, 45, 15], 'kernel': (2, 1, 1),
                'stride': 1, 'groups': 15, 'acivation': nn.ReLU
            }
        ],
        # spatial conv layers: [in, out, (D, H, W), stride]
        [
            # {'in': 15, 'out': 64, 'kernel': (1, 3, 3), 'stride': 1}
        ],
        # for each ConvRNN cell:
        [

        ],
        # temporal convolution stack(s)
        [
            {
                'in': 15, 'out': [128, 256, 128], 'kernel': (2, 3, 3),
                'stride': 1, 'groups': 1, 'acivation': nn.ReLU
            }
        ],
        # ConvTranspose layers: [in, out, (D, H, W), stride]
        [
            {'in': 128, 'out': 64, 'kernel': (3, 3, 3), 'stride': (2, 2, 2)},
            {'in': 64, 'out': 16, 'kernel': (3, 3, 3), 'stride': (1, 2, 2)},
        ],
        # post conv layers
        [
            {'in': 16, 'out': 8, 'kernel': (1, 3, 3), 'stride': 1},
            {'in': 8, 'out': 1, 'kernel': (1, 1, 1), 'stride': 1}
        ],
    )
    return decoder


def decoder_setup_2():
    "This setup was the first big success, solid base config to work from."
    decoder = RetinaDecoder(
        # grouped temporal conv stacks:
        [
            {
                'in': 15, 'out': [45, 45, 15], 'kernel': (2, 1, 1),
                'stride': 1, 'groups': 15, 'acivation': nn.ReLU
            }
        ],
        # spatial conv layers: [in, out, (D, H, W), stride]
        [

        ],
        # for each ConvRNN cell:
        [

        ],
        # temporal convolution stack(s)
        [
            {
                'in': 15, 'out': [128, 256, 128], 'kernel': (2, 3, 3),
                'stride': 1, 'groups': 1, 'acivation': nn.ReLU
            }
        ],
        # ConvTranspose layers: [in, out, (D, H, W), stride]
        [
            {'in': 128, 'out': 64, 'kernel': (3, 3, 3), 'stride': (2, 2, 2)},
            {'in': 64, 'out': 1, 'kernel': (3, 3, 3), 'stride': (1, 2, 2)},
        ],
        # post conv layers
        [

        ],
    )
    return decoder


def decoder_setup_3():
    "Token Conv RNN build."
    decoder = RetinaDecoder(
        # grouped temporal conv stacks:
        # [in, [block_channel(s)], (D, H, W), stride, dilation, groups]
        [
            {
                'in': 15, 'out': [45, 45, 15], 'kernel': (2, 1, 1),
                'stride': 1, 'groups': 15, 'acivation': nn.ReLU
            }
        ],
        # spatial conv layers: [in, out, (D, H, W), stride]
        [
            # {'in': 15, 'out': 64, 'kernel': (2, 1, 1), 'stride': 1}
        ],
        # for each ConvRNN cell:
        [
            {
                'cell': crnns.ConvGRUCell_wnorm, 'dims': (25, 25),
                'in_kernel': (3, 3), 'out_kernel': (3, 3), 'in': 15, 'out': 64,
                'learn_initial': False
            }
        ],
        # temporal convolution stack(s)
        [

        ],
        # ConvTranspose layers: [in, out, (D, H, W), stride]
        [
            {'in': 64, 'out': 64, 'kernel': (3, 3, 3), 'stride': (2, 2, 2)},
            {'in': 64, 'out': 1, 'kernel': (3, 3, 3), 'stride': (1, 2, 2)},
        ],
        # post conv layers
        [

        ],
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
        train_set, test_set, lr=1e-2, epochs=10, batch_sz=4, print_every=80,
        loss_alpha=10
    )

    print('Training set examples...')
    decoder.decode(train_set)
    print('Validation set examples...')
    decoder.decode(test_set)

    if 'n' not in input("Save training set decodings? (y/n): "):
        decoder.save_decodings(train_set)
    if 'n' not in input("Save validation set decodings? (y/n): "):
        decoder.save_decodings(test_set)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    main()
