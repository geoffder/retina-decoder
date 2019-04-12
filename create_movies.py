import numpy as np
from PIL import Image

import os
import h5py as h5


def single_cell_movie(basepath, netdir, recdir, dims):
    # cell locations, shapes, and recordings
    coords = np.loadtxt(basepath+netdir+'cellCoords.csv', delimiter=',')
    # diams = np.loadtxt(basepath+netdir+'cellDiams.csv', delimiter=',')
    recs = np.loadtxt(basepath+netdir+recdir+'cellRecs.csv', delimiter=',')

    movie = np.zeros((*dims, recs.shape[0]))  # .astype(np.float)
    x, y = np.ogrid[:dims[0], :dims[1]]
    for i in range(coords.shape[0]):
        cx, cy = coords[i, :]
        r2 = (x - cx)**2 + (y - cy)**2
        # r = (diams[i]/2)**2
        r = (20/2)**2
        movie[r2 <= r, :] += recs[:, i]
    h5f = h5.File(basepath+netdir+recdir+'cell_movie.h5', 'w')
    h5f.create_dataset('dataset_1', data=movie, compression="gzip")


def build_cell_movie(folder, dims, coords, diams):
    recs = np.loadtxt(folder+'cellRecs.csv', delimiter=',')

    movie = np.zeros((*dims, recs.shape[0]))
    x, y = np.ogrid[:dims[0], :dims[1]]
    for i in range(coords.shape[0]):
        cx, cy = coords[i, :]
        r2 = (x - cx)**2 + (y - cy)**2
        # r = (diams[i]/2)**2
        r = (20/2)**2
        movie[r2 <= r, :] += recs[:, i]

    return recs, movie


def build_stim_movie(folder, dims):
    # decide how I want to be storing stim recordings (note there can be
    # multiple stims, but only one movie)
    pass


def package_experiment(folder, exp_name):
    pckg = h5.File(folder+exp_name+'.h5', 'w')
    # get all entries in given folder that are directories
    net_names = [name for name in os.listdir(folder)
                 if os.path.isdir(folder+name)]
    for net in net_names:
        netgrp = pckg.create_group(net)
        # (x, y) coordinates of all cells in network
        coords = np.loadtxt(folder+net+'/cellCoords.csv', delimiter=',')
        # somas with transparent RFs (just for display)
        net_view = np.loadtxt(folder+net+'/cellMat.csv', delimiter=',')
        # TODO: types and characteristics of the cells
        # store in network group
        netgrp.create_dataset(
            'cell_coords', data=coords, compression='gzip')
        netgrp.create_dataset(
            'net_view', data=net_view, compression='gzip')

        # now move through all stimuli shown to this network
        stim_names = [name for name in os.listdir(folder+net)
                      if os.path.isdir(folder+net+'/'+name)]
        for stim in stim_names:
            stimgrp = netgrp.create_group(stim)
            pth = folder + '/' + net + '/' + stim + '/'
            # TODO: get dims and diams from files
            cell_recs, cell_movie = build_cell_movie(
                                        pth, [600, 600], coords, [])
            # stim_recs, stim_movie = build_stim_movie(pth, [600, 600])
            stimgrp.create_dataset(
                'cell_recs', data=cell_recs, compression="gzip")
            stimgrp.create_dataset(
                'cell_movie', data=cell_movie, compression="gzip")

    pckg.close()


def single_movie_giffer(file):
    # vid = np.load(file+'.npy')
    h5f = h5.File(file+'.h5', 'r')
    vid = h5f['dataset_1'][:]
    vid = vid.transpose(2, 0, 1)
    # normalize and save as gif
    vid = (vid/vid.max()*255).astype(np.uint8)
    # vid = (vid*25.5).astype(np.int8)
    frames = [
        Image.fromarray(vid[i*10]) for i in range(int(vid.shape[0]/10))]
    frames[0].save(file+'.gif', save_all=True, append_images=frames[1:],
                   duration=40, loop=0)


if __name__ == '__main__':
    datapath = 'D:/retina-sim-data/'

    # single_cell_movie(datapath, 'net0/', 'bar0/', [600, 600])
    # single_movie_giffer(datapath+'net0/bar0/cell_movie')

    package_experiment(datapath, 'testExperiment')
