retina-decoder

cnmf modules are taken from the CaImAn toolbox:
https://github.com/flatironinstitute/CaImAn
Isolated to allow easy use of calcium de-convolution functions on 1d waves.
(without having to install the whole toolbox)

To compile cnmf_oasis.pyx, run:
python setup.py build_ext --inplace

You will probably need Visual Studio Build Tools installed (for C++ compiler)
