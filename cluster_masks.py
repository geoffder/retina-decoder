import numpy as np
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA

import os
from cell_clustering import simple_metrics


def get_clusters(root_dir, nets):
    # prepare data (load recordings, extract metrics, and normalize over cells)
    net_list = [simple_metrics(root_dir+net+'/') for net in nets]
    net_pops = [net.shape[0] for net in net_list]  # populations of networks
    data = np.concatenate(net_list, axis=0)
    data = (data - data.mean(axis=0)) / data.std(axis=0)  # normalize (cells)
    print('cell metrics shape (NxD):', data.shape)

    # principle component analysis (get reduced data representation)
    pca = PCA()
    reduced = pca.fit_transform(data)
    # OPTICS clustering
    opticlust = OPTICS(
        min_samples=int(data.shape[0]*.03),  # min neighbours for core points
        # min_cluster_size=.04,
    ).fit(reduced[:, :])  # can cut out "noise" dims  if wanted here
    opt_labels = opticlust.labels_
    print('number of OPTICS clusters:', np.max(opt_labels)+1)

    # break cluster labels back into networks
    cmp = list(np.cumsum(net_pops, dtype=np.int))  # cumlative populations
    net_labels = [opt_labels[start:end] for start, end in zip([0]+cmp, cmp)]

    # group cells (collect indices) by cluster within each network
    net_grps = [
        {
            lbl: [idx for idx, match in enumerate(net == lbl) if match]
            for lbl in set(opt_labels)
        }
        for net in net_labels
    ]

    return net_grps


def sum_masks(pth, folder, clusts):
    masks = np.load(pth+folder+'/masks/all_masks.npy')
    clust_masks = [
        np.sum(masks[idxs], axis=0).clip(0, 1)
        for idxs in clusts.values()
    ]
    return np.stack(clust_masks, axis=0)


def main():
    basepath = 'D:/retina-sim-data/second/'
    datapath = 'D:/retina-sim-data/second/video_dataset/'

    net_names = [
        name for name in os.listdir(basepath)
        if os.path.isdir(basepath+name) and 'net' in name
    ]
    clustered_nets = get_clusters(basepath, net_names)

    clustered_net_masks = [
        sum_masks(datapath, name, net)
        for name, net in zip(net_names, clustered_nets)
    ]

    for name, masks in zip(net_names, clustered_net_masks):
        np.save(datapath+name+'/masks/clusters', masks)


if __name__ == '__main__':
    main()
