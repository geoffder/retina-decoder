import numpy as np
from scipy.signal import resample
from PIL import Image

import os
import h5py as h5
import sqlite3 as sql

# import pandas as pd
import json
from sim_util import rotate


class ProgressBar(object):
    def __init__(self, steps, size=50):
        self.steps = steps
        self.size = size if steps > size else steps
        self.tick = np.floor(steps/self.size)
        self.prog = 0
        self.update()

    def step(self):
        self.prog += 1
        if self.prog % self.tick == 0:
            self.update()
        self.check()

    def update(self):
        ticks = int(np.floor(self.prog/self.tick))
        print(
            '[' + '='*ticks + ' '*(self.size-ticks) + ']',
            end='\r', flush=True
        )

    def check(self):
        if self.prog == self.steps:
            print('')  # new-line


def single_cell_movie(basepath, netdir, recdir, dims):
    # cell locations, shapes, and recordings
    coords = np.loadtxt(basepath+netdir+'cellCoords.csv', delimiter=',')
    diams = [json.loads(line)['diam']
             for line in open(basepath+netdir+'cellParams.txt').readlines()]
    recs = np.loadtxt(basepath+netdir+recdir+'cellRecs.csv', delimiter=',')

    movie = np.zeros((*dims, recs.shape[0]))
    x, y = np.ogrid[:dims[0], :dims[1]]
    for i in range(coords.shape[0]):
        cx, cy = coords[i, :]
        r2 = (x - cx)**2 + (y - cy)**2
        r = (diams[i]/2)**2
        movie[r2 <= r, :] += recs[:, i]
    h5f = h5.File(basepath+netdir+recdir+'cell_movie.h5', 'w')
    h5f.create_dataset('dataset_1', data=movie, compression="gzip")


def get_masks(dims, coords, diams, space_redux):
    # spatial downsampling
    dims = np.array(dims) // space_redux  # must be int for shape
    coords, diams = (coords / space_redux, np.array(diams) / space_redux)

    # stack of masks, shape:(NxHxW)
    masks = np.zeros((coords.shape[0], *dims))
    x, y = np.ogrid[:dims[0], :dims[1]]
    for i in range(coords.shape[0]):
        cx, cy = coords[i, :]
        r2 = (x - cx)**2 + (y - cy)**2
        r = (diams[i]/2)**2
        masks[i, r2 <= r] = 1

    return masks


def build_cell_movie(folder, dims, coords, diams, downsample, space_redux):
    recs = np.loadtxt(folder+'cellRecs.csv', delimiter=',')

    # temporal and spatial downsampling
    duration = recs.shape[0] // downsample
    dims = np.array(dims) // space_redux  # must be int for shape
    coords, diams = (coords / space_redux, np.array(diams) / space_redux)

    movie = np.zeros((duration, *dims))
    x, y = np.ogrid[:dims[0], :dims[1]]
    for i in range(coords.shape[0]):
        cx, cy = coords[i, :]
        r2 = (x - cx)**2 + (y - cy)**2
        r = (diams[i]/2)**2
        movie[:, r2 <= r] += resample(recs[:, i], duration).reshape(-1, 1)

    return recs, movie


def build_stim_movie(folder, dims, downsample, space_redux):
    """
    Take all stimulus recordings and parameters and build a single movie for
    this trial.

    Stim recordings from C++ simulator are saved with the following format:
        (each row) [xpos, ypos, amplitude, orientation]

    Stimulus parameters are saved in text files in JSON parseable format.
    """
    numStims = 0
    for name in os.listdir(folder):
        numStims += 1 if 'stimRecs' in name else 0
    recs = [np.loadtxt(folder+'stimRecs%d.csv' % i, delimiter=',')
            for i in range(numStims)]
    params = [json.loads(open(folder+'stimParams%d.txt' % i).read())
              for i in range(numStims)]

    # temporal and spatial downsampling
    duration = recs[0].shape[0] // downsample
    idx = np.arange(0, recs[0].shape[0], downsample)  # for downsampling slice
    dims = np.array(dims) // space_redux

    movie = np.zeros((duration, *dims))
    x, y = np.ogrid[:dims[0], :dims[1]]
    for rec, param in zip(recs, params):
        # scipy resample was not working corectly for this coordinate data
        # try to figure out why and maybe swtich back. Important for stim
        # and cell recordings to match.
        down_rec = rec[idx]  # simple slice downsampling
        for t in range(duration):
            xpos, ypos, amp, orient = down_rec[t, :]
            xpos, ypos = (xpos / space_redux, ypos / space_redux)
            if param['type'] == 'bar':
                # rotate to match orientation of bar
                xrot, yrot = rotate((xpos, ypos), x, y, np.radians(orient))
                # rectangular mask
                movie[t, :, :] += (
                    (np.abs(xrot-xpos) <= param['width']/2/space_redux)
                    * (np.abs(yrot-ypos) <= param['length']/2/space_redux)
                ) * amp
            elif param['type'] == 'circle':
                # convert cartesian --> polar coordinates
                r2 = (x - xpos)**2 + (y - ypos)**2
                # circular mask
                movie[t, :, :] += (r2 <= (param['radius']/space_redux)**2)*amp

    return recs, movie


def package_experiment(folder, exp_name, downsample=1, space_redux=1):
    """
    Consolidate recordings of cells and stimulus positions (and generated
    movies) in to hdf5 files. For each experiment, the stimuli are the same
    across all networks, thus stimulus recordings and movies are stored in a
    separate hdf5 (saving time and space).
    """
    # connect to sql database
    db = sql.connect(folder+exp_name+'_params.db')
    cursor = db.cursor()
    # create table for network parameters
    cursor.execute(
        '''
        CREATE TABLE NET_PARAMS
        (netstr Text, xdim real, ydim real, margin real, tstop real, dt real)
        '''
    )
    # create table for cell parameters
    cursor.execute(
        '''
        CREATE TABLE CELL_PARAMS
        (
            netstr Text, type Text, diam real, rf_rad real, rf_ax0, rf_ax1,
            dtau real
        )
        '''
    )

    # create hdf5 archive for network recordings
    net_pckg = h5.File(folder+exp_name+'_nets.h5', 'w')
    # get all entries in given folder that are directories
    net_names = [name for name in os.listdir(folder)
                 if os.path.isdir(folder+name) and 'net' in name]
    for net in net_names:
        netgrp = net_pckg.create_group(net)
        # (x, y) coordinates of all cells in network
        coords = np.loadtxt(folder+net+'/cellCoords.csv', delimiter=',')
        # net parameters
        netpars = json.loads(open(folder+net+'/netParams.txt').read())
        cursor.execute(
            'INSERT INTO ' + 'NET_PARAMS' + ' VALUES (?, ?, ?, ?, ?, ?)',
            [net, netpars['xdim'], netpars['ydim'], netpars['margin'],
             netpars['tstop'], netpars['dt']]
        )
        # cell parameters
        params = [json.loads(line)
                  for line in open(folder+net+'/cellParams.txt').readlines()]
        diams = [cell['diam'] for cell in params]
        for par in params:
            cursor.execute(
                'INSERT INTO '+'CELL_PARAMS'+' VALUES (?, ?, ?, ?, ?, ?, ?)',
                [
                    net, par['type'], par['diam'], par.get('rf_rad', 0),
                    par.get('rf_ax0', 0),  par.get('rf_ax1', 0), par['dtau']
                ]
            )
        # somas with transparent RFs (just for display)
        net_view = np.loadtxt(folder+net+'/cellMat.csv', delimiter=',')
        # soma masks of each cell, shape:(NxHxW)
        soma_masks = get_masks(
            [netpars['xdim'], netpars['ydim']], coords, diams, space_redux
        )
        # store in network group
        netgrp.create_dataset(
            'cell_coords', data=coords, compression='gzip')
        netgrp.create_dataset(
            'cell_masks', data=soma_masks, compression='gzip')
        netgrp.create_dataset(
            'net_view', data=net_view, compression='gzip')

        # now move through all stimuli shown to this network
        stim_names = [name for name in os.listdir(folder+net)
                      if os.path.isdir(folder+net+'/'+name)]
        print('Packaging movies for %s...' % net)
        progress = ProgressBar(len(stim_names))
        for stim in stim_names:
            stimgrp = netgrp.create_group(stim)
            pth = folder + '/' + net + '/' + stim + '/'
            cell_recs, cell_movie = build_cell_movie(
                pth, [netpars['xdim'], netpars['ydim']], coords, diams,
                downsample, space_redux
            )
            # store in hdf5
            stimgrp.create_dataset(
                'recs', data=cell_recs, compression="gzip")
            stimgrp.create_dataset(
                'movie', data=cell_movie, compression="gzip")
            progress.step()
    net_pckg.close()

    # create table for network parameters
    cursor.execute(
        '''
        CREATE TABLE STIM_PARAMS
        (stimstr Text, type Text, radius real, width real, length real)
        '''
    )
    # Store stimulus recordings and movies, use last net folder and stim_names
    stim_pckg = h5.File(folder+exp_name+'_stims.h5', 'w')
    print('Packaging movies for stimuli...')
    progress = ProgressBar(len(stim_names))
    for stim in stim_names:
        pth = folder + '/' + net + '/' + stim + '/'
        # consolidate stimulus parameters into SQL table
        fnames = [name for name in os.listdir(pth)
                  if os.path.isfile(pth+name) and 'stimParams' in name]
        for f in fnames:
            p = json.loads(open(pth+f).read())
            cursor.execute(
                    'INSERT INTO ' + 'STIM_PARAMS' + ' VALUES (?, ?, ?, ?, ?)',
                    [stim, p['type'], p['radius'], p['width'], p['length']]
                )
        # create and store stimulus movies
        stimgrp = stim_pckg.create_group(stim)
        stim_recs, stim_movie = build_stim_movie(
            pth, [netpars['xdim'], netpars['ydim']], downsample, space_redux
        )
        stimgrp.create_dataset(
            'recs', data=stim_recs, compression="gzip")
        stimgrp.create_dataset(
            'movie', data=stim_movie, compression="gzip")
        progress.step()
    stim_pckg.close()

    db.commit()
    db.close()


def movie_giffer(fname, vid, max_val=None, downsample=1):
    """
    Takes desired filename (without '.gif') and a numpy matrix
    (in H x W x Frames organization right now) and saves it as a GIF using
    the PIL.Image module.
    """
    # normalize and save as gif
    if max_val is None:
        vid = (vid/vid.max()*255).clip(0, 255).astype(np.uint8)
    else:
        vid = (vid/max_val*255).clip(0, 255).astype(np.uint8)
    frames = [
        Image.fromarray(vid[i*downsample], mode='P')
        for i in range(int(vid.shape[0]/downsample))
    ]
    frames[0].save(fname+'.tif', save_all=True, append_images=frames[1:],
                   duration=40, loop=0, optimize=False)


def test_gifs(stim):
    stim_names = [stim+'%i' % d for d in [0, 45, 90, 135, 180, 225, 270, 315]]
    # stim_names = ['bar%i' % d for d in [0, 45, 90, 135, 180]]

    # extract and gif the network movies
    net_pckg = h5.File(datapath+'testExperiment_nets.h5', 'r')
    nets = [net_pckg['net0'][name]['movie'][:] for name in stim_names]

    print('Giffing network movies...')
    max_val = np.max([net.max() for net in nets])  # for normalization
    progress = ProgressBar(len(nets))
    for name, net in zip(stim_names, nets):
        movie_giffer(datapath+'net_'+name, net, max_val)
        progress.step()
    del net_pckg, nets  # free up the memory

    # extract and gif the stim movies
    stim_pckg = h5.File(datapath+'testExperiment_stims.h5', 'r')
    stims = [stim_pckg[name]['movie'][:] for name in stim_names]
    print('Giffing stimuli movies...')
    progress = ProgressBar(len(stims))
    for name, stim in zip(stim_names, stims):
        movie_giffer(datapath+'stim_'+name, stim)
        progress.step()


def build_folder_dataset(basepath, folder, downsample=1, space_redux=1):
    # create dataset parent folder (basepath is where raw recordings are)
    datafolder = basepath+folder
    os.mkdir(datafolder)

    # get all entries in given folder that are directories
    net_names = [
        name for name in os.listdir(basepath)
        if os.path.isdir(basepath+name) and 'net' in name
    ]
    # all stimuli shown to these networks
    stim_names = [
        name for name in os.listdir(basepath+net_names[0])
        if os.path.isdir(basepath+net_names[0]+'/'+name)
    ]
    for net in net_names:
        # new folder for this network's movies
        os.mkdir(datafolder+net)
        # (x, y) coordinates of all cells in network
        coords = np.loadtxt(basepath+net+'/cellCoords.csv', delimiter=',')
        # net parameters
        netpars = json.loads(open(basepath+net+'/netParams.txt').read())
        # cell parameters
        params = [json.loads(line)
                  for line in open(basepath+net+'/cellParams.txt').readlines()]
        diams = [cell['diam'] for cell in params]
        # soma masks of each cell, shape:(NxHxW)
        soma_masks = get_masks(
            [netpars['xdim'], netpars['ydim']], coords, diams, space_redux
        )
        # make folder and store masks of all cells
        os.mkdir(datafolder+net+'/masks/')
        np.save(datafolder+net+'/masks/all_masks', soma_masks)

        # create and store videos of cells
        os.mkdir(datafolder+net+'/cells/')
        print('Packaging movies for %s...' % net)
        progress = ProgressBar(len(stim_names))
        for stim in stim_names:
            pth = basepath + '/' + net + '/' + stim + '/'
            _, cell_movie = build_cell_movie(
                pth, [netpars['xdim'], netpars['ydim']], coords, diams,
                downsample, space_redux
            )
            np.save(datafolder+net+'/cells/'+stim, cell_movie)
            progress.step()

    os.mkdir(datafolder+'/stims/')
    print('Packaging movies for stimuli...')
    progress = ProgressBar(len(stim_names))
    for stim in stim_names:
        pth = basepath + '/' + net_names[0] + '/' + stim + '/'
        _, stim_movie = build_stim_movie(
            pth, [netpars['xdim'], netpars['ydim']], downsample, space_redux
        )
        np.save(datafolder+'/stims/'+stim, stim_movie)
        progress.step()


if __name__ == '__main__':
    datapath = 'D:/retina-sim-data/'

    # package_experiment(
    #     datapath, 'testExperiment', downsample=10, space_redux=4
    # )
    # test_gifs('med_light_bar')

    build_folder_dataset(
        datapath, 'video_dataset/', downsample=10, space_redux=4
    )