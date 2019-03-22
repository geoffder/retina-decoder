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


def plotRecs(recTable_file):
    "Plot recording of each cell for whole 'experiment'"
    matrix = pd.read_csv(recTable_file, header=None).values
    nrows = 10
    ncols = matrix.shape[1]//nrows
    ncols += 1 if matrix.shape[1] % nrows > 0 else 0
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 9))
    for i in range(matrix.shape[1]):
        axes[i % nrows, i//nrows].plot(
            np.arange(matrix.shape[0])*1, matrix[:, i])
        axes[i % nrows, i//nrows].set_ylabel('c%s' % i, rotation=0)
    fig.tight_layout()
    plt.show()


data_path = "D://work/"
plotNet(data_path+'cellCoords.csv', data_path+'cellMat.csv')
plot2D(data_path+'stimMask.csv')
plotRecs(data_path+'cellRecs.csv')
