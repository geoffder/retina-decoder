import os

from suite2p_wrapper import get_suite2p_data, get_raw_scans, get_beams
from suite2p_wrapper import create_masks, pack_hdf


def main():
    datapath = os.getcwd()
    suitepath = os.path.join(datapath, 'suite2p', 'plane0')

    recs, neu, stats = get_suite2p_data(suitepath)
    print("Recordings for %d cells loaded." % recs.shape[0])

    # Load in 2PLSM scans (.tif) and stack them.
    mov = get_raw_scans(datapath)
    # Extract mean Z-projections of each cell from raw movie using their ROIs.
    beams = get_beams(mov, stats)

    # Create binary spatial masks for each cell using ypix and xpix indices.
    masks = create_masks(stats, mov.shape[1:])

    # Save suite2p outputs, and raw signal (extracted with rois) to HDF5 file
    pack_hdf(datapath, 'suite2pack', recs, neu, beams, masks)
    print("ROIs and signals saved to suite2pack.h5 in %s" % datapath)


if __name__ == '__main__':
    main()
