import numpy as np
import pandas as pd

import os

import torch
from torch.utils.data import Dataset


class RetinaVideos(Dataset):
    """Cluster encoded ganglion network and stimulus video Dataset."""

    def __init__(self, root_dir, preload=False, crop_centre=None):
        self.root_dir = root_dir
        self.rec_frame = self.build_lookup(root_dir)
        self.preload = preload
        self.crop_centre = crop_centre
        if preload:
            self.all_masks, self.all_stims = self.preloader()
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

    def preloader(self):
        "Preload cluster masks and stimuli into RAM for faster DataLoader."
        masks = [self.get_masks(idx) for idx in range(self.__len__())]
        stims = [self.get_stim(idx) for idx in range(self.__len__())]
        if self.crop_centre is not None:
            masks = [self.crop(mask) for mask in masks]
            stims = [self.crop(stim) for stim in stims]
        return masks, stims

    def crop(self, matrix):
        "Take (_, H, W) matrix and crop centre of spatial dimensions."
        ox, oy = np.array(matrix.shape[1:]) // 2
        x, y = np.array(self.crop_centre) // 2
        return matrix[:, ox-x:ox+x, oy-y:oy+y]

    def get_masks(self, idx):
        masks = np.load(os.path.join(
                self.root_dir,
                self.rec_frame.iloc[idx, 0],  # net folder
                'masks',
                'clusters.npy'
            ))
        return masks

    def get_stim(self, idx):
        stim = np.load(os.path.join(
                self.root_dir,
                'stims',
                self.rec_frame.iloc[idx, 1]  # file name
            ))
        return stim

    def __len__(self):
        "Number of samples in dataset."
        return len(self.rec_frame)

    def __getitem__(self, idx):
        # network activity movie
        rec = np.load(os.path.join(
            self.root_dir,
            self.rec_frame.iloc[idx, 0],  # net folder
            'cells',
            self.rec_frame.iloc[idx, 1]  # file name
        ))
        rec = self.crop(rec) if self.crop_centre is not None else rec
        # cluster encoding masks and target stimuli
        if self.preload:
            masks = self.all_masks[idx]
            stim = self.all_stims[idx]
        else:
            if self.crop_centre is not None:
                masks = self.crop(self.get_masks(idx))
                stim = self.crop(self.get_stim(idx))
            else:
                masks = self.get_masks(idx)
                stim = self.get_stim(idx)

        # keep stimulus in -1 to 1 range (max contrasts of black/white)
        stim = stim.clip(-1, 1)

        # Take shape (TxHxW) and encode to (TxCxHxW) with C cluster masks
        rec = np.stack([rec*mask for mask in masks], axis=1)

        # convert to torch Tensors
        rec = torch.from_numpy(rec).float()
        stim = torch.from_numpy(stim).float().unsqueeze(1)  # add channel dim

        sample = {'net': rec, 'stim': stim}
        return sample
