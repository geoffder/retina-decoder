import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from skimage import io

import os


def get_suite2p_data(pth):
    """
    Load extracted recordings (F, Fneu) and cell information (stat, which
    includes ROI pixels.) from the given suite2p output folder.

    recs & recs_neu: (N, T) ndarray
    stats: Numpy object containing of dicts for all N cells.
    """
    # get indices of all accepted cell ROIs
    cellids = np.nonzero(
        np.load(pth + 'iscell.npy')[:, 0].astype(np.int)
    )[0]

    # get extracted fluourence signals for accepted cells only. shape:(N, T)
    recs = np.load(pth + 'F.npy')[cellids, :]
    neu = np.load(pth + 'Fneu.npy')[cellids, :]

    # get stat dict containing ROI masks (etc) for accepted cells
    stats = np.load(pth + 'stat.npy', allow_pickle=True)[cellids]

    return recs, neu, stats


def get_raw_scans(datapath, start_scan, num_scans):
    """
    Load in original 2PLSM scans, stack them and return as (T, Y, X) array.
    """
    mov = np.concatenate([
        io.imread("%sScan_%03d_ch1.tif" % (datapath, num))
        for num in range(start_scan, start_scan+num_scans)
    ], axis=0)
    return mov


def get_beams(movie, stats):
    """
    Pull out z-projection for each cell using it's generated ROI.
    Returns an ndarray of shape (N, T).
    """
    beams = np.array([
        movie[:, cell['ypix'], cell['xpix']].mean(axis=1)
        for cell in stats]
    )
    return beams


def create_masks(stats, dims):

    masks = np.zeros((stats.size, *dims), dtype=np.int8)
    for idx in range(stats.size):
        masks[idx, stats[idx]['ypix'], stats[idx]['xpix']] = 1
    return masks


def pack_hdf(pth, fname, Fcell, Fneu, raw_beams, roi_masks):
    """
    Store recordings and masks in and HDF5 archive. If Igor can't read these,
    then use pandas to_hdf() function, that should work.
    """
    with h5.File(pth + fname + '.hdf5', 'w') as pckg:
        pckg.create_dataset("Fcell", data=Fcell)
        pckg.create_dataset("Fneu", data=Fneu)
        pckg.create_dataset("masks", data=roi_masks)
        pckg.create_dataset("beams", data=raw_beams)


def store_csvs(pth, fldr, Fcell, Fneu, raw_beams, roi_masks):
    """
    Store recordings and masks as CSVs to make loading in to IgorPro easy.
    Save floats with limited precision, and masks are integers for readability
    and storage space considerations.
    """
    savepth = os.path.join(pth, fldr)
    if not os.path.isdir(savepth):
        os.mkdir(savepth)

    np.savetxt(os.path.join(savepth, 'Fcell.csv'), Fcell.T, '%1.4f', ',')
    np.savetxt(os.path.join(savepth, 'Fneu.csv'), Fneu.T, '%1.4f', ',')
    np.savetxt(os.path.join(savepth, 'beams.csv'), raw_beams.T, '%1.4f', ',')

    # Save masks seperately in to a sub-folder
    maskpth = os.path.join(savepth, 'masks')
    if not os.path.isdir(maskpth):
        os.mkdir(maskpth)

    for i, msk in enumerate(roi_masks):
        np.savetxt(os.path.join(maskpth, 'roi%d.csv' % i), msk, '%d', ',')


def plot_signals(Fcell, Fall, raw, idx, norm=False, plot_Fall=False):
    """
    Plot suite2p outputs with mask beam for comparison.
    """
    # pull out cell of interest, normalizing if norm=True
    Fcell_wave = Fcell[idx] / Fcell[idx].max() if norm else Fcell[idx]
    Fall_wave = Fall[idx] / Fall[idx].max() if norm else Fall[idx]
    raw_wave = raw[idx] / raw[idx].max() if norm else raw[idx]

    plt.plot(Fcell_wave, label='F')
    plt.plot(Fall_wave, label='F+Fneu') if plot_Fall else 0
    plt.plot(raw_wave, label='Raw Beam')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    datapath = "/media/geoff/Data/calcium/test_set/"
    suitepath = datapath + "suite2p/plane0/"

    recs, neu, stats = get_suite2p_data(suitepath)
    print("Recordings for %d cells loaded." % recs.shape[0])

    # Load in 2PLSM scans (.tif) and stack them.
    mov = get_raw_scans(datapath, 9, 8)
    # Extract mean Z-projections of each cell from raw movie using their ROIs.
    beams = get_beams(mov, stats)

    # Create binary spatial masks for each cell using ypix and xpix indices.
    masks = create_masks(stats, mov.shape[1:])

    # Save data as CSVs into sub-folder
    # store_csvs(datapath, 'csv_data', recs, neu, beams, masks)

    plot_signals(recs, recs-neu*.8, beams, 10)
