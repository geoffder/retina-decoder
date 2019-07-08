import numpy as np
import matplotlib.pyplot as plt

import os

import recurrent_convolution as crnns
from temporal_convolution import TemporalConv3dStack, CausalTranspose3d
from util_modules import Permuter, make_pool3d_layer
from sim_util import StackPlotter
from create_movies import ProgressBar

import torch
from torch import nn
# import torch.nn.functional as F
from torch import optim

from custom_loss import DecoderLoss
from retina_dataset import RetinaVideos
from torch.utils.data import DataLoader

# import time as timer

"""
Thoughts:
- scale the calculated loss up by a function of how much alpha has
    decreased, since decaying alpha will decrease loss on it's own?
    out = mean(loss) * start_alpha/current_alpha
    There is probably a better equation, but this gets at the idea...

- Tried 5x5x5 transpose (no post-conv) and found the resulting decodings
    were diffuse, may be that the 5x5 spatial part of the kernel was too
    expansive especially at the first spatial upsampling stage.

- try RMSProp with momentum soon, see whether more stable than ADAM

- try using channel grouping deeper in the the network, maybe it does not need
    to mix information across them until the transpose/decoding phase

- batch_sz=8 *seems* to make gradient more stable in training, however,
    even with lr=1e-1 to speed things up, the decodings produced after 20
    epochs seem spread out, blown-up, or deformed somehow. Is it possible that
    the larger batchsize averages together the errors of the different stimuli
    too much, such that weights of each channel can't get dialed in as well?

    TODO: More comparisons of different batch sizes should be done. 4 was the
    old standard before Colab allowed larger sizes. What is the sweet spot?
    Try 1 -> 6 and get a feel for this dynamic.

- before writing off batch_sz=8, try proper 30-40 epoch run, rather than
    running 20 epoch twice (which screws with ADAM and loss alpha)
"""


class RetinaDecoder(nn.Module):

    def __init__(self, pre_pool, grp_tempo_params, conv_params,
                 crnn_cell_params, temp3d_stack_params, decode_params):
        super(RetinaDecoder, self).__init__()
        # layer parameters
        self.pre_pool = pre_pool
        self.grp_tempo_params = grp_tempo_params
        self.conv_params = conv_params
        self.crnn_cell_params = crnn_cell_params
        self.temp3d_stack_params = temp3d_stack_params
        self.decode_params = decode_params
        # create model and send to correct device (GPU if available)
        self.build()
        self.dv = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.dv)

    def build(self):
        # # # # # # # # # # ENCODER NETWORK # # # # # # # # # #
        encoder_mods = []

        # pooling operation before any processing
        if 'op' in self.pre_pool:  # skip by leaving param dict empty
            encoder_mods.append(make_pool3d_layer(self.pre_pool))

        # Grouped Temporal CNN, operating on each cluster channel separately
        for p in self.grp_tempo_params:
            encoder_mods.append(
                TemporalConv3dStack(
                    p['in'], p['out'], p.get('kernel', (2, 1, 1)),
                    p.get('space_dilation', 1), p.get('groups', 1),
                    p.get('dropout', 0), p.get('activation', nn.ReLU)
                )
            )
            if 'pool' in p:
                encoder_mods.append(make_pool3d_layer(p['pool']))

        # Spatial Only (non-causal) convolutional layers
        for p in self.conv_params:
            d, h, w = p.get('kernel', (1, 3, 3))
            pad = (d//2, h//2, w//2)
            encoder_mods.append(
                nn.Conv3d(
                    p['in'], p['out'], (d, h, w), p.get('stride', 1), pad,
                    p.get('dilation', 1), p.get('groups', 1),
                    p.get('bias', True)
                )
            )
            encoder_mods.append(nn.BatchNorm3d(p['out']))
            encoder_mods.append(p.get('activation', nn.ReLU)())
            if 'pool' in p:
                encoder_mods.append(make_pool3d_layer(p['pool']))

        # Stack of Convolutional Recurrent Network(s)
        if len(self.crnn_cell_params) > 0:
            # swap time from depth dimension to first dimension for CRNN(s)
            # (N, C, T, H, W) -> (T, N, C, H, W)
            encoder_mods.append(Permuter((2, 0, 1, 3, 4)))
        for p in self.crnn_cell_params:
            # recurrenct convolutional cells (GRU or LSTM)
            encoder_mods.append(
                p.get('crnn_cell', crnns.ConvGRUCell_wnorm)(
                    p['dims'], p['in_kernel'], p['out_kernel'], p['in'],
                    p['out'], p.get('learn_initial', False),
                    p.get('return_hidden', False)
                )
            )
            if 'post_activation' in p:
                encoder_mods.append(p['post_activation']())
        if len(self.crnn_cell_params) > 0:
            # swap time back to depth dimension following CRNN(s)
            # (T, N, C, H, W) -> (N, C, T, H, W)
            encoder_mods.append(Permuter((1, 2, 0, 3, 4)))

        # Temporal CNN
        for p in self.temp3d_stack_params:
            encoder_mods.append(
                TemporalConv3dStack(
                    p['in'], p['out'], p.get('kernel', (2, 3, 3)),
                    p.get('space_dilation', 1), p.get('groups', 1),
                    p.get('dropout', 0), p.get('activation', nn.ReLU)
                )
            )

        # package encoding layers as a Sequential network
        self.encoder_net = nn.Sequential(*encoder_mods)

        # # # # # # # # # # DECODER NETWORK # # # # # # # # # #
        decoder_mods = []

        # Transpose Convolutional layers (upsampling)
        for p in self.decode_params:
            # unpack kernel etc dimensions
            d, h, w = p.get('kernel', (1, 3, 3))
            st_d, st_h, st_w = p.get('stride', (1, 1, 1))
            dil_d, dil_h, dil_w = p.get('dilation', (1, 1, 1))

            # causal transpose
            if p.get('type', 'causal') == 'causal':
                decoder_mods.append(
                    CausalTranspose3d(
                        p['in'], p['out'], p['kernel'], p['stride'],
                        p.get('groups', 1), p.get('bias', True),
                        p.get('dilations', (1, 1, 1))
                    )
                )
            # non-causal transpose
            elif p['type'] == 'trans':
                pad = (d//2, h//2, w//2)
                decoder_mods.append(
                    nn.ConvTranspose3d(
                        p['in'], p['out'], p['kernel'], p['stride'],
                        pad, pad, p.get('groups', 1), p.get('bias', True),
                        p.get('dilations', (1, 1, 1))
                    )
                )
            # plain convolution (spatial only) -> p['type'] == 'conv'
            else:
                pad = (d//2, h//2, w//2)
                decoder_mods.append(
                    nn.Conv3d(
                        p['in'], p['out'], (d, h, w), p.get('stride', 1), pad,
                        p.get('dilation', 1), p.get('groups', 1),
                        p.get('bias', True)
                    )
                )

            decoder_mods.append(nn.BatchNorm3d(p['out']))
            decoder_mods.append(p.get('activation', nn.Tanh)())

        # package decoding layers as a Sequential network
        self.decoder_net = nn.Sequential(*decoder_mods)

    def forward(self, X):
        X = self.encoder_net(X)
        X = self.decoder_net(X)
        return X

    def fit(self, train_set, test_set, lr=1e-4, epochs=10, batch_sz=1,
            loss_alpha=10, loss_decay=1, print_every=0, peons=2):

        train_loader = DataLoader(
            train_set, batch_size=batch_sz, shuffle=True, num_workers=peons
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_sz, num_workers=peons
        )
        N = train_set.__len__()  # number of samples

        # DecoderLoss equivalent to MSE when alpha=0 (original default: 10)
        self.loss = DecoderLoss(alpha=loss_alpha, decay=loss_decay).to(self.dv)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-8)

        n_batches = np.ceil(N / batch_sz).astype('int')
        print_every = n_batches if print_every < 1 else print_every
        train_prog = None
        train_costs, test_costs = [], []
        for i in range(epochs):
            cost = 0
            print("epoch:", i, "n_batches:", n_batches)
            # start = 0
            for j, batch in enumerate(train_loader):
                # print('time to load batch', timer.time()-start)
                # start = timer.time()
                net, stim = batch['net'].to(self.dv), batch['stim'].to(self.dv)
                cost += self.train_step(net, stim)
                del net, stim, batch
                # print('time to train', timer.time()-start)
                train_prog.step() if train_prog is not None else 0
                if j % print_every == 0:
                    test_prog = ProgressBar(
                        np.ceil(test_set.__len__() / batch_sz).astype('int'),
                        size=np.ceil(
                            test_set.__len__() / batch_sz
                        ).astype('int'),
                        label='validating: '
                    )
                    # costs and accuracies for test set
                    test_cost = 0
                    for t, testB in enumerate(test_loader, 1):
                        net = testB['net'].to(self.dv)
                        stim = testB['stim'].to(self.dv)
                        testB_cost = self.get_cost(net, stim)
                        del net, stim, testB
                        test_cost += testB_cost
                        test_prog.step()
                    test_cost /= t+1
                    print("validation cost: %f" % (test_cost))

                    train_prog = ProgressBar(
                        print_every, size=test_set.__len__()*2 // batch_sz,
                        label='training:   '
                    )
                    train_prog.step() if j == 0 else 0  # hack, skipped batch
                # start = timer.time()

            # Decay DecoderLoss sparsity penalty
            self.loss.decay()

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
        decoded = self.forward(inputs)  # (N, C, T, H, W)
        output = self.loss.forward(
            # swap time to second dimension -> (N, T, C, H, W)
            decoded.transpose(1, 2), targets
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
            decoded = self.forward(inputs)  # (N, C, T, H, W)
            output = self.loss.forward(
                # swap time to second dimension -> (N, T, C, H, W)
                decoded.transpose(1, 2), targets
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
                net = sample['net'].to(self.dv)
                decoded = self.forward(net)
                del net

            # Reduce out batch and channel dims, then put time last
            # (N, C, T, H, W) -> (H, W, T)
            decoded = decoded.squeeze().cpu().numpy().transpose(1, 2, 0)
            net = sample['net'].squeeze().numpy().sum(axis=0)
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

    def save_decodings(self, sample_set, name=None):
        self.eval()  # set the model to testing mode
        sample_loader = DataLoader(sample_set, batch_size=1, num_workers=2)

        # make a parent output folder for this dataset if it doesn't exist
        outfold = os.path.join(sample_set.root_dir, 'outputs')
        if not os.path.isdir(outfold):
            os.mkdir(outfold)
        # prompt for name of and create this particular runs output folder
        while True:
            nametag = input("Decoding set name: ") if name is None else name
            name = None  # if parameter name fails, get input next loop
            basefold = os.path.join(outfold, nametag)
            if not os.path.isdir(basefold):
                os.mkdir(basefold)
                break
            else:
                print('Folder exists, provide another name...')

        # generate decoding of every sample in given dataset
        for i, sample in enumerate(sample_loader):
            with torch.no_grad():
                # get stimulus prediction from network activity
                net = sample['net'].to(self.dv)
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
            # .npy format
            np.save(
                # file name corresponding to stimulus
                os.path.join(decofold, sample_set.rec_frame.iloc[i, 1]),
                decoded
            )


def decoder_setup_1():
    """
    This is the same as setup_2 in decoder_v4, which was the first breakthrough
    network, except here the pooling operations have been set to causal mode.
    On colab, 2x 20 epochs with lr=1e-1 and batch_sz=8 has produced strong
    strong decoding results.

    Best run ever:
    lr=1e-2, batch_sz=6, loss_alpha=10, loss_decay=.9 (ran 30 epochs)
    Consider running with even more epochs, was consistently decreasing the
    whole time, only bounced up on the last two epochs.
    epoch 27 -> .007085; 28 -> .006431; 29 -> .007603
    file: nopost_batch6_lre-2_epoch30.modl and corresponding outputs folder
    """
    decoder = RetinaDecoder(
        # pre-pooling
        {'op': 'avg', 'kernel': (1, 2, 2), 'causal': True},
        # grouped temporal conv stacks:
        [
            {
                'in': 14, 'out': [45, 45, 15], 'kernel': (2, 1, 1),
                'stride': 1, 'groups': 14, 'acivation': nn.ReLU,
                'pool': {'op': 'avg', 'kernel': (2, 2, 2), 'causal': True}
            }
        ],
        # spatial conv layers: {in, out, kernel, stride}
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
        # Decoder network layers: 'type': -> 'causal' || 'trans' || 'conv'
        # can interleave (causal)transpose and regular convolutions
        [
            {
                'type': 'causal', 'in': 128, 'out': 64, 'kernel': (3, 3, 3),
                'stride': (2, 2, 2),
            },
            {
                'type': 'causal', 'in': 64, 'out': 1, 'kernel': (3, 3, 3),
                'stride': (1, 2, 2),
            }
        ],
    )
    return decoder


def decoder_setup_2():
    "Post-conv, same as setup_1 in decoder_v4. Copied over, might delete."
    decoder = RetinaDecoder(
        # pre-pooling
        {'op': 'avg', 'kernel': (1, 2, 2), 'causal': True},
        # grouped temporal conv stacks:
        [
            {
                'in': 15, 'out': [45, 45, 15], 'kernel': (2, 1, 1),
                'stride': 1, 'groups': 15, 'acivation': nn.ReLU,
                'pool': {'op': 'avg', 'kernel': (2, 2, 2), 'causal': True}
            }
        ],
        # spatial conv layers: {in, out, kernel, stride}
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
        # Decoder network layers: 'type': -> 'causal' || 'trans' || 'conv'
        # can interleave (causal)transpose and regular convolutions
        [
            {
                'type': 'causal', 'in': 128, 'out': 64, 'kernel': (3, 3, 3),
                'stride': (2, 2, 2),
            },
            {
                'type': 'causal', 'in': 64, 'out': 16, 'kernel': (3, 3, 3),
                'stride': (1, 2, 2),
            },
            {'type': 'conv', 'in': 16, 'out': 8, 'kernel': (1, 3, 3)},
            {'type': 'conv', 'in': 8, 'out': 1, 'kernel': (1, 3, 3)},
        ],
    )
    return decoder


def decoder_setup_3():
    """
    Experiment with inter-leaving causal transpose convolutions and plain
    spatial-only convolutions.
    """
    decoder = RetinaDecoder(
        # pre-pooling
        {'op': 'avg', 'kernel': (1, 2, 2), 'causal': True},
        # grouped temporal conv stacks:
        [
            {
                'in': 15, 'out': [45, 45, 15], 'kernel': (2, 1, 1),
                'stride': 1, 'groups': 15, 'acivation': nn.ReLU,
                'pool': {'op': 'avg', 'kernel': (2, 2, 2), 'causal': True}
            }
        ],
        # spatial conv layers: {in, out, kernel, stride}
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
        # Decoder network layers: 'type': -> 'causal' || 'trans' || 'conv'
        # can interleave (causal)transpose and regular convolutions
        [
            {
                'type': 'causal', 'in': 128, 'out': 64, 'kernel': (3, 3, 3),
                'stride': (2, 2, 2),
            },
            {'type': 'conv', 'in': 64, 'out': 64, 'kernel': (1, 3, 3)},
            {
                'type': 'causal', 'in': 64, 'out': 16, 'kernel': (3, 3, 3),
                'stride': (1, 2, 2),
            },
            {'type': 'conv', 'in': 16, 'out': 8, 'kernel': (1, 3, 3)},
            {'type': 'conv', 'in': 8, 'out': 1, 'kernel': (1, 3, 3)},
        ],
    )
    return decoder


def main():
    if os.name == 'posix':
        basepath = '/media/geoff/Data/retina-sim-data/third/'
    else:
        basepath = 'D:/retina-sim-data/third/'

    train_path = basepath + 'train_video_dataset/'
    test_path = basepath + 'test_video_dataset/'

    print('Building datasets...')
    train_set = RetinaVideos(
        train_path, preload=False, crop_centre=[100, 100], time_first=False,
        frame_lag=0
    )
    test_set = RetinaVideos(
        test_path, preload=False, crop_centre=[100, 100], time_first=False,
        frame_lag=0
    )

    # testing hdf5 dataset structure
    # train_set = RetinaVideos(
    #     basepath, 'train_video_dataset.h5', preload=False,
    #     crop_centre=[100, 100], time_first=False, frame_lag=0
    # )
    # test_set = RetinaVideos(
    #     basepath, 'test_video_dataset.h5', preload=False,
    #     crop_centre=[100, 100], time_first=False, frame_lag=0
    # )

    print('Building model...')
    decoder = decoder_setup_2()

    if 'n' not in input("Load pre-trained state dict? (y/n):"):
        while True:
            dict_path = input("Path to pickled RetinaDecoder state_dict:")
            if not os.path.isfile(dict_path):
                print('Not a file, typo? Try again...')
            else:
                break
        decoder.load_state_dict(torch.load(dict_path))

    print('Fitting model...')
    decoder.fit(
        train_set, test_set, lr=1e-2, epochs=20, batch_sz=4, print_every=0,
        loss_alpha=10, loss_decay=.9, peons=2
    )

    print('Training set examples...')
    decoder.decode(train_set)
    print('Validation set examples...')
    decoder.decode(test_set)

    if 'n' not in input("Save trained decoder's state dict? (y/n):"):
        path = input('Name for pickled model state:')
        torch.save(decoder.state_dict(), path)

    if 'n' not in input("Save training set decodings? (y/n):"):
        decoder.save_decodings(train_set)
    if 'n' not in input("Save validation set decodings? (y/n):"):
        decoder.save_decodings(test_set)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    main()
