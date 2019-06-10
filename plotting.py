import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plotNet(coords_file, matrix_file):
    coords = pd.read_csv(coords_file, header=None).values
    matrix = pd.read_csv(matrix_file, header=None).values

    fig, ax = plt.subplots(1)
    ax.imshow(matrix)
    for i in range(coords.shape[0]):
        ax.scatter(coords[i, 0], coords[i, 1], c='r', alpha=.5, s=80,
                   marker='$%s$' % i)
    plt.show()


def plot2D(matrix_file):
    matrix = pd.read_csv(matrix_file, header=None).values

    fig, ax = plt.subplots(1)
    ax.imshow(matrix)
    plt.show()


def plotRecs(net_path, stim_names, fname, cells=None):
    "Plot recording of every cell (or those in list) for specified stims."
    recs = np.concatenate([
        pd.read_csv(net_path+stim+fname, header=None).values
        for stim in stim_names
    ], axis=0)
    cells = [i for i in range(recs.shape[1])] if cells is None else cells
    recs = recs[:, cells][np.arange(0, len(recs), 10)]  # downsample
    recs /= recs.max(axis=0)  # normalize cells to themselves

    nrows = 10 if cells is None or len(cells) > 10 else len(cells)
    ncols = recs.shape[1]//nrows
    ncols += 1 if recs.shape[1] % nrows > 0 else 0
    # if ncols > 1:
    #     fig, axes = plt.subplots(nrows, ncols, figsize=(19, 9))
    # else:
    #     fig, axes = plt.subplots(nrows, figsize=(19, 9))
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 9))
    for i in range(recs.shape[1]):
        axes[i % nrows, i//nrows].plot(
            np.arange(recs.shape[0])*1, recs[:, i])
        axes[i % nrows, i//nrows].set_ylabel('c%s' % i, rotation=0)
    fig.tight_layout()
    plt.show()


data_path = "D:/retina-sim-data/second/"
net_name = 'net0'
plotNet(
    data_path+net_name+'/cellCoords.csv',
    data_path+net_name+'/cellMat.csv'
)
# plot2D(data_path+'stimMask.csv')

stims = ['med_light_bar'+str(i*45) for i in range(8)]
cells = [i+2 for i in range(20)]
plotRecs(data_path+net_name+'/', stims, '/cellRecs.csv', cells=cells)
