import os
from pathlib import Path
import subprocess
import sys

from setuptools import setup
from setuptools.extension import Extension
import numpy as np


if os.getenv('MAXVOLPY_CYTHON', '1').lower() in {'1', 'true', 'yes', 'on'}:
    maxvol_pyx = Path(__file__).parent.joinpath('maxvolpy', '_maxvol.pyx')
    maxvol_pyx_src = maxvol_pyx.with_suffix(maxvol_pyx.suffix + '.src')

    if not maxvol_pyx.is_file() or maxvol_pyx_src.stat().st_mtime > maxvol_pyx.stat().st_mtime:
        subprocess.call([sys.executable, maxvol_pyx_src], cwd=maxvol_pyx_src.parent)

    ext_modules = [
        Extension(
            'maxvolpy._maxvol',
            ['maxvolpy/_maxvol.pyx'],
            include_dirs=[np.get_include()],
            extra_compile_args=['-O3', '-march=native', '-ffast-math'],
        ),
    ]
else:
    ext_modules = None


setup(ext_modules=ext_modules)
