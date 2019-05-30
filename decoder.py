import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from recurrent_convolution import ConvGRUCell, ConvLSTMCell
from sim_util import StackPlotter

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

"""
Thoughts:
- should the (1x1) dimension reduction convolution happen in a recurrent cell,
    or should it be done with a plain Conv2D layer after recurrence is over?
- getting infinite loss when stacking CRNN cells. Need to normalize between?
    Seems like an exploding gradient issue.

- now I'm thinking the (1x1) kernel dimension reduction (to 1 channel) should
    happen frame by frame (2d conv) after the final recurrence.
- now loss isn't inf/nan with clipping/sigmoid, but now the decoding is
    ceilinged out. Need to read into batchnorm in RNNs, and at least try
    the easy addition of norm after recurrent layers.
"""


class RetinaDecoder(nn.Module):

    def __init__(self, dims, crnn_cell_params, crnn_cell=ConvGRUCell,
                 learn_initial=False):
        super(RetinaDecoder, self).__init__()
        self.dims = dims
        self.crnn_cell_params = crnn_cell_params
        self.crnn_cell = crnn_cell
        self.learn_initial = learn_initial
        self.build()
        self.to(device)

    def build(self):
        self.crnn_stack = nn.ModuleList()
        for i, params in enumerate(self.crnn_cell_params):
            # recurrenct convolutional cells (GRU or LSTM)
            self.crnn_stack.append(
                self.crnn_cell(
                    self.dims, *params, learn_initial=self.learn_initial
                )
            )

    def forward(self, X):
        for cell in self.crnn_stack:
            X, _ = cell(X)
            # X = F.elu(X)
            X = torch.sigmoid(X)
        X = torch.clamp(X, 0, 1)
        return X

    def fit(self, train_set, test_set, lr=1e-4, epochs=40, batch_sz=10,
            print_every=15):

        train_loader = DataLoader(
            train_set, batch_size=batch_sz, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_sz, num_workers=2
        )
        N = train_set.__len__()  # number of samples

        self.loss = nn.MSELoss().to(device)
        # self.loss = nn.BCELoss().to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        n_batches = N // batch_sz
        train_costs, test_costs = [], []
        for i in range(epochs):
            cost = 0
            print("epoch:", i, "n_batches:", n_batches)
            for j, batch in enumerate(train_loader):

                cost += self.train_step(
                    batch['net'].transpose(0, 1).to(device),
                    batch['stim'].to(device)
                )

                if j % print_every == 0:
                    # costs and accuracies for test set
                    test_cost = 0
                    for t, testB in enumerate(test_loader, 1):
                        testB_cost = self.get_cost(
                            testB['net'].transpose(0, 1).to(device),
                            testB['stim'].to(device)
                        )
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

            # Reduce out batch and channel dims (T, N, C, H, W) -> (T, H, W)
            decoded = decoded.squeeze().cpu().numpy().transpose(1, 2, 0)
            net = sample['net'].squeeze().numpy().sum(axis=1)
            net = net.transpose(1, 2, 0)
            stim = sample['stim'].squeeze().numpy().transpose(1, 2, 0)

            # synced scrollable videos of cell actity, decoding, and stimulus
            fig, ax = plt.subplots(1, 3)
            net_stack = StackPlotter(ax[0], net, delta=1)
            deco_stack = StackPlotter(ax[1], decoded, delta=1)
            stim_stack = StackPlotter(ax[2], stim, delta=1)
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


class RetinaVideos(Dataset):
    """Cluster encoded ganglion network and stimulus video Dataset."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.rec_frame = self.build_lookup(root_dir)
        # maybe load the cluster masks for all networks and keep in mem?
        # same for stimuli? Just load in the network recs since they are the
        # bigger lump of data

    @staticmethod
    def build_lookup(root_dir):
        """
        Make a lookup table of recordings (file names correspond to stimuli)
        found in the replicate network folders. Corresponding recording and
        stimuli files are found in the 'root/net#/cells/' and 'root/stims/'
        folders respectively.
        """
        net_names = [
            name for name in os.listdir(root_dir)
            if os.path.isdir(root_dir+name) and 'net' in name
        ]
        table = [
            [net, file]
            for k, net in enumerate(net_names)
            for file in [name for name in os.listdir(root_dir+net+'/cells/')
                         if os.path.isfile(root_dir+net+'/cells/'+name)]
        ]
        return pd.DataFrame(table)

    def __len__(self):
        "Number of samples in dataset."
        return len(self.rec_frame)

    def __getitem__(self, idx):
        rec = np.load(os.path.join(
            self.root_dir,
            self.rec_frame.iloc[idx, 0],  # net folder
            'cells',
            self.rec_frame.iloc[idx, 1]  # file name
        ))
        masks = np.load(os.path.join(
            self.root_dir,
            self.rec_frame.iloc[idx, 0],  # net folder
            'masks',
            'clusters.npy'
        ))
        stim = np.load(os.path.join(
            self.root_dir,
            'stims',
            self.rec_frame.iloc[idx, 1]  # file name
        ))

        # normalize movies (shouldn't be doing this sample by sample)
        # sort out some kind of a batch norm scheme
        rec = (rec - rec.mean()) / rec.std()
        stim = (stim - stim.min()) / (np.abs(stim.min())+stim.max())
        # Take shape (TxHxW) and encode to (TxCxHxW) with C cluster masks
        rec = np.stack([rec*mask for mask in masks], axis=1)

        # convert to torch Tensors
        rec = torch.from_numpy(rec).float()
        stim = torch.from_numpy(stim).float().unsqueeze(1)  # add channel dim

        sample = {'net': rec, 'stim': stim}
        return sample


def decoder_setup_1():
    decoder = RetinaDecoder(
        (175, 175),  # input spatial dimensions
        # for each cell: [in_kernel, out_kernel, in_channels, out_channels]
        [
            [(3, 3), (3, 3), 14, 28],
            [(3, 3), (3, 3), 28, 56],
            [(3, 3), (3, 3), 56, 28],
            [(3, 3), (1, 1), 56, 1]
        ],
        crnn_cell=ConvLSTMCell
    )
    return decoder


def decoder_setup_2():
    decoder = RetinaDecoder(
        (175, 175),  # input spatial dimensions
        # for each cell: [in_kernel, out_kernel, in_channels, out_channels]
        [
            [(13, 13), (7, 7), 14, 28],
            [(3, 3), (3, 3), 28, 1],
        ],
        crnn_cell=ConvGRUCell
    )
    return decoder


def decoder_setup_3():
    decoder = RetinaDecoder(
        (175, 175),  # input spatial dimensions
        # for each cell: [in_kernel, out_kernel, in_channels, out_channels]
        [
            [(3, 3), (3, 3), 14, 1],
        ],
        crnn_cell=ConvGRUCell
    )
    return decoder


def main():
    train_path = 'D:/retina-sim-data/video_dataset/'
    test_path = 'D:/retina-sim-data/small_video_dataset/'

    train_set = RetinaVideos(test_path)
    test_set = RetinaVideos(test_path)

    decoder = decoder_setup_2()
    decoder.fit(
        train_set, test_set, lr=1e-5, epochs=1, batch_sz=1, print_every=3
    )
    decoder.decode(train_set)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    main()
