import numpy as np

import os
import h5py as h5


def h5_packer(videopath):
    # create hdf5 archive
    vid_pckg = h5.File(videopath[:-1]+'.h5', 'w')

    # get all entries in given folder that are directories
    folders = [
        name for name in os.listdir(videopath)
        if os.path.isdir(videopath+name) and 'net' in name
    ] + ['stims']

    for fldr in folders:
        grp = vid_pckg.create_group(fldr)
        sub_folders = [
            sub for sub in os.listdir(videopath+fldr)
            if os.path.isdir(os.path.join(videopath, fldr, sub))
        ]
        sub_folders = [''] if len(sub_folders) < 1 else sub_folders
        for sub in sub_folders:
            subgrp = grp.create_group(sub) if len(sub) > 0 else grp
            for file in os.listdir(os.path.join(videopath, fldr, sub)):
                subgrp.create_dataset(
                    file,
                    data=np.load(os.path.join(videopath, fldr, sub, file))
                )

    vid_pckg.close()


if __name__ == '__main__':
    datapath = '/media/geoff/Data/retina-sim-data/third/'

    h5_packer(datapath+'train_video_dataset/')
