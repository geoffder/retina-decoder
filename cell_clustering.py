import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

import os
import json

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import MeanShift, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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


def hierarchical(data):
    methods = ['ward', 'single', 'complete', 'average', 'weighted']

    for method in methods:
        metric = 'correlation' if method != 'ward' else 'euclidean'
        Z = linkage(data, method=method, metric=metric, optimal_ordering=True)
        dendrogram(Z)
        plt.title('method: %s, metric: %s' % (method, metric))
        plt.show()


if __name__ == '__main__':
    # change colours used for sequential plotting
    steps = 15
    new_colors = [plt.get_cmap('jet')(1. * i/steps) for i in range(steps)]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors)))

    # folder names
    datapath = 'D:/retina-sim-data/'
    net_names = [name for name in os.listdir(datapath)
                 if os.path.isdir(datapath+name) and 'net' in name]

    # prepare data (load recordings, extract metrics, and normalize over cells)
    data = np.concatenate([
        simple_metrics(datapath+net+'/') for net in net_names
    ], axis=0)
    data = (data - data.mean(axis=0)) / data.std(axis=0)  # normalize (cells)
    print('cell metrics shape (NxD):', data.shape)

    cell_params = [
        json.loads(line)
        for net in net_names
        for line in open(datapath+net+'/'+'cellParams.txt').readlines()
    ]
    cell_types = [
        prm["type"] + str(prm.get("theta", ''))
        + ('Sust' if prm.get("sustained", 0) else 'Trans'
            if 'Alpha' in prm['type'] else '')
        for prm in cell_params
    ]
    print('Number of Cell Types:', len(set(cell_types)))

    # look-ups, mapping cell types to indices for grouping and plotting
    type2ind = {}
    ind2type = []
    type_inds = []
    ind = 0
    for cell in cell_types:
        if cell not in type2ind:
            type2ind[cell] = ind
            ind2type.append(cell)
            ind += 1
        type_inds.append(type2ind[cell])
    type_inds = np.array(type_inds)

    # principle component analysis
    pca = PCA()
    reduced = pca.fit_transform(data)

    # mean-shift clustering
    clustering = MeanShift().fit(reduced[:, :2])
    ms_labels = clustering.labels_
    print('number of MeanShift clusters:', np.max(ms_labels)+1)
    # convert cell indices to types to view cluster -> cell type mapping
    ms_groups = {
        lbl: [ind2type[ind] for ind in type_inds[ms_labels == lbl]]
        for lbl in set(ms_labels)
    }

    # OPTICS clustering
    opticlust = OPTICS(
        min_samples=int(data.shape[0]*.03),  # min neighbours for core points
        # min_cluster_size=.04,
    ).fit(reduced[:, :])
    opt_labels = opticlust.labels_
    print('number of OPTICS clusters:', np.max(opt_labels)+1)
    # convert cell indices to types to view cluster -> cell type mapping
    opt_groups = {
        lbl: [ind2type[ind] for ind in type_inds[opt_labels == lbl]]
        for lbl in set(opt_labels)
    }
    # print out clusters
    for grp in opt_groups.items():
        print('group %d size: %d' % (grp[0], len(grp[1])))
        print(grp)

    # Top 2 PCA componenets, plotted with cell type Labels
    fig1, ax1 = plt.subplots(1, 3, figsize=(14, 6))
    ax1[0].scatter(
        reduced[:, 0], reduced[:, 1], alpha=.5, s=100, c=type_inds, cmap='jet'
    )
    ax1[0].set_title('with Cell Type Labels')
    ax1[0].set_xlabel('component 1')
    ax1[0].set_ylabel('component 2')
    # Top 2 PCA componenets, plotted with Mean Shift Labels
    ax1[1].scatter(
        reduced[:, 0], reduced[:, 1], alpha=.5, s=100, c=opt_labels, cmap='jet'
    )
    ax1[1].set_title('with OPTICS Labels')
    ax1[1].set_xlabel('component 1')
    ax1[1].set_ylabel('component 2')
    # cumulative variance explained
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    ax1[2].plot(cumulative)
    ax1[2].set_title('Cumulative Information')
    ax1[2].set_xlabel('dimensions')
    ax1[2].set_ylabel('variance explained')
    fig1.tight_layout()

    # embed with t-SNE into 2 dimensions
    X_embed2d = TSNE(
        n_components=2, perplexity=30).fit_transform(reduced[:, :])
    # plot the 2D embedding of the 2D data
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6))
    ax2[0].scatter(
        X_embed2d[:, 0], X_embed2d[:, 1], c=type_inds, alpha=.5, s=100,
        cmap='jet'
    )
    # arbitrarily place text label on first instance of each type in dataset
    for ind in range(len(ind2type)):
        row = np.where(type_inds == ind)[0][0]  # take the first one
        ax2[0].scatter(
            X_embed2d[row, 0], X_embed2d[row, 1], alpha=.5, s=6000, c='c',
            marker='$%s$' % ind2type[ind]
        )
    ax2[0].set_title('TSNE 2D Embedding with Cell Type labels')
    ax2[0].set_xlabel('dimension 1')
    ax2[0].set_ylabel('dimension 2')
    ax2[1].scatter(
        X_embed2d[:, 0], X_embed2d[:, 1], c=opt_labels, alpha=.5, s=100,
        cmap='jet'
    )
    ax2[1].set_title('TSNE 2D Embedding with OPTICS labels')
    ax2[1].set_xlabel('dimension 1')
    ax2[1].set_ylabel('dimension 2')

    plt.show()

    # hierarchical(reduced[:, :10])
