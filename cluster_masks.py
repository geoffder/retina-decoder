import numpy as np
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA

import os
from cell_clustering import simple_metrics


def get_clusters(root_dir, nets, ignored_stims=[]):
    # prepare data (load recordings, extract metrics, and normalize over cells)
    net_list = [
        simple_metrics(root_dir+net+'/', ignored_stims=ignored_stims)
        for net in nets
    ]
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


def create_cluster_masks(basepath, videopath, ignore):
    net_names = [
        name for name in os.listdir(basepath)
        if os.path.isdir(basepath+name) and 'net' in name
    ]
    clustered_nets = get_clusters(basepath, net_names, ignored_stims=ignore)

    clustered_net_masks = [
        sum_masks(videopath, name, net)
        for name, net in zip(net_names, clustered_nets)
    ]

    for name, masks in zip(net_names, clustered_net_masks):
        np.save(videopath+name+'/masks/clusters', masks)


def main():
    if os.name == 'posix':
        basepath = '/media/geoff/Data/retina-sim-data/'
    else:
        basepath = 'D:/retina-sim-data/'

    basepath += 'fourth/'
    datapath = basepath + 'video_dataset/'

    # only want full field stimuli (long bars)
    ignore = ['circle', 'collision']

    # cluster cells, combine ROI masks, and save
    create_cluster_masks(basepath, datapath, ignore)


if __name__ == '__main__':
    main()
