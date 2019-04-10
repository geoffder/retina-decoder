import numpy as np
import pandas as pd
import h5py
from PIL import Image


def cell_movie(basepath, netdir, recdir, dims):
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
    movie = (movie*1000).astype(np.int8)
    # np.save(basepath+netdir+recdir+'cell_movie', movie)
    h5f = h5py.File(basepath+netdir+recdir+'cell_movie.h5', 'w')
    h5f.create_dataset('dataset_1', data=movie, compression="gzip")


def movie_giffer(file):
    # vid = np.load(file+'.npy')
    h5f = h5py.File(file+'.h5', 'r')
    vid = h5f['dataset_1'][:]
    vid = vid.transpose(2, 0, 1)
    # normalize and save as gif
    vid = (vid/vid.max()*255).astype(np.int8)
    # vid = (vid*25.5).astype(np.int8)
    frames = [
        Image.fromarray(vid[i*10]) for i in range(int(vid.shape[0]/10))]
    frames[0].save(file+'.gif', save_all=True, append_images=frames[1:],
                   duration=40)


if __name__ == '__main__':
    datapath = 'D:/retina-sim-data/'
    # cell_movie(datapath, 'net0/', 'bar0/', [600, 600])
    movie_giffer(datapath+'net0/bar0/cell_movie')
