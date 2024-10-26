import os

from setuptools import setup
from setuptools.extension import Extension
import numpy as np


if os.name != 'nt' and os.getenv('MAXVOLPY_CYTHON', '1').lower() in {'1', 'true', 'yes', 'on'}:
    ext_modules = [
        Extension(
            'maxvolpy._maxvol',
            ['src/maxvolpy/_maxvol.pyx'],
            include_dirs=[np.get_include()],
            extra_compile_args=['-O3', '-march=native', '-ffast-math', '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION']
        ),
    ]
else:
    ext_modules = None


setup(ext_modules=ext_modules)
