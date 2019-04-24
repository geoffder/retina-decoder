import numpy as np
from PIL import Image

import os
import h5py as h5
import sqlite3 as sql

# import pandas as pd
import json
from sim_util import rotate


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


def build_cell_movie(folder, dims, coords, diams):
    recs = np.loadtxt(folder+'cellRecs.csv', delimiter=',')

    movie = np.zeros((*dims, recs.shape[0]))
    x, y = np.ogrid[:dims[0], :dims[1]]
    for i in range(coords.shape[0]):
        cx, cy = coords[i, :]
        r2 = (x - cx)**2 + (y - cy)**2
        r = (diams[i]/2)**2
        movie[r2 <= r, :] += recs[:, i]

    return recs, movie


def build_stim_movie(folder, dims):
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

    movie = np.zeros((*dims, recs[0].shape[0]))
    x, y = np.ogrid[:dims[0], :dims[1]]
    for rec, param in zip(recs, params):
        for t in range(rec.shape[0]):
            xpos, ypos, amp, orient = rec[t, :]
            if param['type'] == 'bar':
                # rotate to match orientation of bar
                xrot, yrot = rotate((xpos, ypos), x, y, np.radians(orient))
                # rectangular mask
                movie[:, :, t] += (
                    (np.abs(x-xpos) <= param['width'])
                    * (np.abs(y-ypos) <= param['length'])
                ) * amp
            elif param['type'] == 'circle':
                # convert cartesian --> polar coordinates
                r2 = (x - xpos)**2 + (y - ypos)**2
                # circular mask
                movie[:, :, t] += (r2 <= param['radius']**2) * amp

    return recs, movie


def package_experiment(folder, exp_name):
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
        (netstr Text, type Text, diam real, rf_rad real, dtau real)
        '''
    )

    # create hdf5 archive for network recordings
    net_pckg = h5.File(folder+exp_name+'_nets.h5', 'w')
    # get all entries in given folder that are directories
    net_names = [name for name in os.listdir(folder)
                 if os.path.isdir(folder+name)]
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
                'INSERT INTO ' + 'CELL_PARAMS' + ' VALUES (?, ?, ?, ?, ?)',
                [net, par['type'], par['diam'], par['rf_rad'], par['dtau']]
            )
        # somas with transparent RFs (just for display)
        net_view = np.loadtxt(folder+net+'/cellMat.csv', delimiter=',')
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
            cell_recs, cell_movie = build_cell_movie(
                pth, [netpars['xdim'], netpars['ydim']], coords, diams)
            # store in hdf5
            stimgrp.create_dataset(
                'recs', data=cell_recs, compression="gzip")
            stimgrp.create_dataset(
                'movie', data=cell_movie, compression="gzip")
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
            pth, [netpars['xdim'], netpars['ydim']])
        stimgrp.create_dataset(
            'recs', data=stim_recs, compression="gzip")
        stimgrp.create_dataset(
            'movie', data=stim_movie, compression="gzip")
    stim_pckg.close()

    db.commit()
    db.close()


def movie_giffer(fname, matrix):
    """
    Takes desired filename (without '.gif') and a numpy matrix
    (in H x W x Frames organization right now) and saves it as a GIF using
    the PIL.Image module.
    """
    vid = matrix.transpose(2, 0, 1)
    # normalize and save as gif
    vid = (vid/vid.max()*255).astype(np.uint8)
    frames = [
        Image.fromarray(vid[i*10]) for i in range(int(vid.shape[0]/10))]
    frames[0].save(fname+'.gif', save_all=True, append_images=frames[1:],
                   duration=40, loop=0)


if __name__ == '__main__':
    datapath = 'D:/retina-sim-data/'

    package_experiment(datapath, 'testExperiment')
