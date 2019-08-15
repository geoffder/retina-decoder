import numpy as np
import matplotlib.pyplot as plt

from skimage import io

if __name__ == '__main__':
    datapath = "/media/geoff/Data/calcium/test_set/"
    suitepath = datapath + "suite2p/plane0/"

    # get indices of all accepted cell ROIs
    cellids = np.nonzero(
        np.load(suitepath + 'iscell.npy')[:, 0].astype(np.int)
    )[0]

    # get extracted fluourence signals for accepted cells only. shape:(N, T)
    recs = np.load(suitepath + 'F.npy')[cellids, :]
    neu = np.load(suitepath + 'Fneu.npy')[cellids, :]
    recs_neu = recs + neu

    # get stat dict containing ROI masks (etc) for accepted cells
    stats = np.load(suitepath + 'stat.npy', allow_pickle=True)[cellids]

    print("Recordings for %d cells loaded." % recs.shape[0])

    # test out signal extraction with masks vs recs from suite2p
    mov = np.concatenate([
        io.imread(datapath + 'Scan_' + num + '_ch1.tif')
        for num in ['009', '010', '011', '012', '013', '014', '015', '016']
    ], axis=0)

    beams = np.array([
        mov[:, cell['ypix'], cell['xpix']].sum(axis=1)
        for cell in stats
    ])

    # plot normalized suite2p outputs with mask beam for comparison
    plt.plot(recs[0] / recs[0].max(), label='F')
    plt.plot(recs_neu[0] / recs_neu[0].max(), label='F+Fneu')
    plt.plot(beams[0] / beams[0].max(), label='Raw Beam')
    plt.legend()
    plt.show()
