import numpy as np
import pandas as pd

import os

import torch
from torch.utils.data import Dataset


class RetinaVideos(Dataset):
    """Cluster encoded ganglion network and stimulus video Dataset."""

    def __init__(self, root_dir, preload=False, crop_centre=None,
                 time_first=True, frame_lag=0):
        self.root_dir = root_dir
        self.rec_frame = self.build_lookup(root_dir)
        self.preload = preload
        self.crop_centre = crop_centre
        self.time_first = time_first  # whether time is first dim, or depth
        self.frame_lag = frame_lag
        if preload:  # load data that is common across samples to memory
            self.all_masks, self.all_stims = self.preloader()

        self.mmaps = self.create_mmaps()

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

    def create_mmaps(self):
        recs = [self.map_recs(idx) for idx in range(self.__len__())]
        masks = [self.map_masks(idx) for idx in range(self.__len__())]
        stims = [self.map_stims(idx) for idx in range(self.__len__())]
        return {'recs': recs, 'stims': stims, 'masks': masks}

    def preloader(self):
        "Preload cluster masks and stimuli into RAM for faster DataLoader."
        masks = [self.get_masks(idx) for idx in range(self.__len__())]
        stims = [self.get_stim(idx) for idx in range(self.__len__())]
        return masks, stims

    def crop(self, matrix):
        "Take (_, H, W) matrix and crop centre of spatial dimensions."
        ox, oy = np.array(matrix.shape[1:]) // 2
        x, y = np.array(self.crop_centre) // 2
        return matrix[:, ox-x:ox+x, oy-y:oy+y]

    def map_masks(self, idx):
        "Load cluster masks for each network into memory."
        masks = np.load(os.path.join(
                self.root_dir,
                self.rec_frame.iloc[idx, 0],  # net folder
                'masks',
                'clusters.npy'
            ),
            mmap_mode='r'
        )
        return masks

    def map_stims(self, idx):
        "Load stimuli (common across all networks) into memory."
        stim = np.load(os.path.join(
                self.root_dir,
                'stims',
                self.rec_frame.iloc[idx, 1]  # file name
            ),
            mmap_mode='r'
        )[self.frame_lag:, :, :]  # skip photo-receptor lag frames
        # return self.crop(stim) if self.crop_centre is not None else stim
        return stim

    def map_recs(self, idx):
        "Load network activity recording into memory."
        rec = np.load(os.path.join(
                self.root_dir,
                self.rec_frame.iloc[idx, 0],  # net folderget
                'cells',
                self.rec_frame.iloc[idx, 1]  # file name
            ),
            mmap_mode='r'
        )
        rec = rec[:-self.frame_lag, :, :] if self.frame_lag > 0 else rec
        # return self.crop(rec) if self.crop_centre is not None else rec
        return rec

    def __len__(self):
        "Number of samples in dataset."
        return len(self.rec_frame)

    def __getitem__(self, idx):
        # network activity movie
        # rec = self.get_rec(idx)
        # cluster encoding masks and target stimuli
        # if self.preload:
        #     masks = self.all_masks[idx]
        #     stim = self.all_stims[idx]
        # else:
        #     masks = self.get_masks(idx)
        #     stim = self.get_stim(idx)

        rec, stim, masks = [
            self.mmaps[k][idx] for k in ['recs', 'stims', 'masks']
        ]

        # keep stimulus in -1 to 1 range (max contrasts of black/white)
        stim = stim.clip(-1, 1)

        # Take shape (TxHxW) and encode with C cluster masks to (TxCxHxW) if
        # time_first, or (CxTxHxW) if not
        channel_dim = 1 if self.time_first else 0
        rec = np.stack([rec*mask for mask in masks], axis=channel_dim)

        # convert to torch Tensors
        rec = torch.from_numpy(rec).float()
        stim = torch.from_numpy(stim).float().unsqueeze(1)  # add channel dim

        sample = {'net': rec, 'stim': stim}
        return sample
