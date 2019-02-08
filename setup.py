from setuptools import setup, find_packages
import sys
import numpy as np
from Cython.Build import cythonize
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext

if sys.platform == 'darwin':
    extra_compiler_args = ['-stdlib=libc++']
else:
    extra_compiler_args = []

ext_modules = [Extension("cnmf_oasis",
                         sources=["cnmf_oasis.pyx"],
                         include_dirs=[np.get_include()],
                         language="c++",
                         extra_compile_args=extra_compiler_args)]

setup(
    packages=find_packages(exclude=['use_cases', 'use_cases.*']),
    install_requires=[''],
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext}
)
