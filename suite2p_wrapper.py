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
        np.load(os.path.join(pth, 'iscell.npy'))[:, 0].astype(np.int)
    )[0]

    # get extracted fluourence signals for accepted cells only. shape:(N, T)
    recs = np.load(os.path.join(pth, 'F.npy'))[cellids, :]
    neu = np.load(os.path.join(pth, 'Fneu.npy'))[cellids, :]

    # get stat dict containing ROI masks (etc) for accepted cells
    stats = np.load(os.path.join(pth, 'stat.npy'), allow_pickle=True)[cellids]

    return recs, neu, stats


def get_raw_scans(datapath, prefix='Scan', start_scan=False, num_scans=False):
    """
    Load in original 2PLSM scans, stack them and return as (T, Y, X) array.
    If start_scan and num_scans are not specified, all scans in the folder are
    loaded.
    """
    if not (start_scan and num_scans):
        fnames = [
            os.path.join(datapath, f)
            for f in os.listdir(datapath)
            if os.path.isfile(os.path.join(datapath, f)) and '_ch' in f
        ]
    else:
        fnames = [
            "%s%s_%03d_ch1.tif" % (datapath, prefix, num)
            for num in range(start_scan, start_scan+num_scans)
        ]

    return np.concatenate([io.imread(fname) for fname in fnames], axis=0)


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
    """
    Use ypix and xpix index arrays to generate binary (0 || 1) spatial masks
    for each cell, and return as a 3d int8 ndarray of shape (N, Y, X).
    """
    masks = np.zeros((stats.size, *dims), dtype=np.int8)
    for idx in range(stats.size):
        masks[idx, stats[idx]['ypix'], stats[idx]['xpix']] = 1
    return masks


def pack_hdf(pth, fname, Fcell, Fneu, raw_beams, roi_masks):
    """
    Store recordings and masks in an HDF5 archive using h5py. Extracted
    recordings and transposed to shape:(Time, Cell), and mask stack dimensions
    are permuted to shape:(X, Y, Cell) to align with IgorPro formatting.
    """
    with h5.File(pth + fname + '.h5', 'w') as pckg:
        pckg.create_dataset("Fcell", data=Fcell.T)
        pckg.create_dataset("Fneu", data=Fneu.T)
        pckg.create_dataset("masks", data=roi_masks.transpose(2, 1, 0))
        pckg.create_dataset("beams", data=raw_beams.T)


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
    mov = get_raw_scans(datapath, 'Scan')

    # Extract mean Z-projections of each cell from raw movie using their ROIs.
    beams = get_beams(mov, stats)

    # Create binary spatial masks for each cell using ypix and xpix indices.
    masks = create_masks(stats, mov.shape[1:])

    # Save suite2p outputs, and raw signal (extracted with rois) to HDF5 file
    pack_hdf(datapath, 'suite2pack', recs, neu, beams, masks)

    plot_signals(recs, recs-neu*.8, beams, 10)
