import numpy as np
import pandas as pd

import os
import h5py as h5

import torch
from torch.utils.data import Dataset

import time as timer


class RetinaVideos(Dataset):
    """Cluster encoded ganglion network and stimulus video Dataset."""

    def __init__(self, root_dir, h5_name, preload=False, crop_centre=None,
                 time_first=True, frame_lag=0):
        self.root_dir = root_dir
        self.h5_name = h5_name
        self.h5_path = os.path.join(root_dir, h5_name)
        self.rec_frame = self.build_lookup(self.h5_path)
        # load data that is common across samples to memory
        self.preload = preload
        if preload:
            self.all_masks, self.all_stims = self.preloader()
        # transformations
        self.crop_centre = crop_centre
        self.time_first = time_first  # whether time is first dim, or depth
        self.frame_lag = frame_lag

    @staticmethod
    def build_lookup(path):
        """
        Make a lookup table of recordings (file names correspond to stimuli)
        found in the replicate network groups. Corresponding recording and
        stimuli files are found in the '/net#/cells/' and '/stims/' groups
        respectively.
        """
        with h5.File(path, 'r') as pckg:
            table = [
                [net, file]
                for net in pckg.keys() if 'net' in net  # avoid 'stims' group
                for file in pckg[net]['cells'].keys()
            ]
        return pd.DataFrame(table)

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

    def get_masks(self, idx):
        "Load cluster masks for each network into memory."
        with h5.File(self.h5_path, 'r') as pckg:
            net_grp = self.rec_frame.iloc[idx, 0]
            masks = pckg[net_grp]['masks']['clusters.npy'][:]
        return self.crop(masks) if self.crop_centre is not None else masks

    def get_stim(self, idx):
        "Load stimuli (common across all networks) into memory."
        with h5.File(self.h5_path, 'r') as pckg:
            stim = pckg['stims'][self.rec_frame.iloc[idx, 1]][:]
            stim = stim[self.frame_lag:, :, :] if self.frame_lag > 0 else stim
        return self.crop(stim) if self.crop_centre is not None else stim

    def get_rec(self, idx):
        "Load network activity recording into memory."
        net_grp = self.rec_frame.iloc[idx, 0]
        fname = self.rec_frame.iloc[idx, 1]
        with h5.File(self.h5_path, 'r') as pckg:
            rec = pckg[net_grp]['cells'][fname][:]
            rec = rec[:-self.frame_lag, :, :] if self.frame_lag > 0 else rec
        return self.crop(rec) if self.crop_centre is not None else rec

    def __len__(self):
        "Number of samples in dataset."
        return len(self.rec_frame)

    def __getitem__(self, idx):
        # start = timer.time()
        # network activity movie
        rec = self.get_rec(idx)
        # cluster encoding masks and target stimuli
        if self.preload:
            masks = self.all_masks[idx]
            stim = self.all_stims[idx]
        else:
            masks = self.get_masks(idx)
            stim = self.get_stim(idx)
        # print('time to get sample', timer.time()-start)

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
