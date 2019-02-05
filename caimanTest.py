import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.io import imread

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches
from caiman.utils.visualization import nb_plot_contour

fnames = ['D:\\calcium\\171030_03_GCAMP6f_tiffs\\Scan_009_ch1.tif']

# a = imread(fnames[0])
# avg = a.mean(axis=0)
# plt.imshow(avg, cmap='gray')
# plt.show()

# dataset dependent parameters
fr = 8  # 30                         # imaging rate in frames per second
decay_time = .5  # 0.4            # length of a typical transient in seconds

# motion correction parameters
strides = (48, 48)
overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
max_shifts = (12, 12)  # (6,6)    # maximum allowed rigid shifts (in pixels)
max_deviation_rigid = 3
pw_rigid = True             # flag for performing non-rigid motion correction

# parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system
gnb = 2                     # number of global background components
merge_thresh = 0.8          # merging threshold, max correlation allowed
rf = 30  # 15            # half-size of the patches in pixels.
stride_cnmf = 12  # 6       # amount of overlap between the patches in pixels
K = 4                       # number of components per patch
gSig = [10, 10]  # [4, 4]            # expected half size of neurons in pixels
method_init = 'greedy_roi'  # initialization method
ssub = 1                    # spatial subsampling during initialization
tsub = 1                    # temporal subsampling during intialization

# parameters for component evaluation
min_SNR = 2.0               # signal to noise ratio for accepting a component
rval_thr = 0.85       # space correlation threshold for accepting a component
cnn_thr = 0.99              # threshold for CNN based classifier
cnn_lowest = 0.1  # neurons with cnn probability lower are rejected

opts_dict = {
    'fnames': fnames,
    'fr': fr,
    'decay_time': decay_time,
    'strides': strides,
    'overlaps': overlaps,
    'max_shifts': max_shifts,
    'max_deviation_rigid': max_deviation_rigid,
    'pw_rigid': pw_rigid,
    'p': 1,
    'nb': gnb,
    'rf': rf,
    'K': K,
    'stride': stride_cnmf,
    'method_init': method_init,
    'rolling_sum': True,
    'only_init': True,
    'ssub': ssub,
    'tsub': tsub,
    'min_SNR': min_SNR,
    'rval_thr': rval_thr,
    'use_cnn': True,
    'min_cnn_thr': cnn_thr,
    'cnn_lowest': cnn_lowest
}
opts = params.CNMFParams(params_dict=opts_dict)

# %% start a cluster for parallel processing (if a cluster already exists it
# will be closed and a new session will be opened)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
# mc.motion_correct(save_movie=True)
# m_els = cm.load(mc.fname_tot_els)
border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0

# memory map the file in order 'C'
fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                           border_to_0=border_to_0)  # exclude borders

# now load the file
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
    #load frames in python format (T x X x Y)
# image = imread(fnames[0])
opts.set('temporal', {'p': 0})
cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit(images)
