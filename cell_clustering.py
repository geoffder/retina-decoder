import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from scipy.cluster.hierarchy import linkage, dendrogram


def simple_metrics(net_dir):
    # all stimuli shown to this network
    stim_names = [name for name in os.listdir(net_dir)
                  if os.path.isdir(net_dir+name)]
    recs = [np.loadtxt(net_dir+stim+'/cellRecs.csv', delimiter=',').T
            for stim in stim_names]

    metrics = np.concatenate([
        np.concatenate([
            np.sum(rec, axis=1, keepdims=True),
            np.max(rec, axis=1, keepdims=True),
            np.max(rec, axis=1, keepdims=True)
            / (np.sum(rec, axis=1, keepdims=True) + .00001),
        ], axis=1)
        for rec in recs
    ], axis=1)

    return metrics


if __name__ == '__main__':
    datapath = 'D:/retina-sim-data/'
    net_names = [name for name in os.listdir(datapath)
                 if os.path.isdir(datapath+name) and 'net' in name]

    data = np.concatenate([
        simple_metrics(datapath+net+'/') for net in net_names
    ], axis=0)
    print('cell metrics shape (NxD):', data.shape)

    Z = linkage(
        data, method='ward', metric='euclidean', optimal_ordering=True
    )
    dendrogram(Z)
    plt.title('ward')
    plt.show()

    Z = linkage(
        data, method='single', metric='correlation', optimal_ordering=True
    )
    dendrogram(Z)
    plt.title('single')
    plt.show()

    Z = linkage(
        data, method='complete', metric='correlation', optimal_ordering=True
    )
    dendrogram(Z)
    plt.title('complete')
    plt.show()

    Z = linkage(
        data, method='average', metric='correlation', optimal_ordering=True
    )
    dendrogram(Z)
    plt.title('average')
    plt.show()

    Z = linkage(
        data, method='weighted', metric='correlation', optimal_ordering=True
    )
    dendrogram(Z)
    plt.title('weighted')
    plt.show()
