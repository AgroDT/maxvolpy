import os

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import numpy as np


class BuildExt(build_ext):
    def build_extension(self, ext) -> None:
        if self.compiler.compiler_type != 'msvc':
            ext.extra_compile_args.extend(('-O3', '-march=native', '-ffast-math'))
        return super().build_extension(ext)


def get_ext_modules():
    if os.getenv('MAXVOLPY_CYTHON', '1').lower() not in {'1', 'true', 'yes', 'on'}:
        return None
    return [
        Extension(
            'maxvolpy._maxvol',
            ['src/maxvolpy/_maxvol.pyx'],
            include_dirs=[np.get_include()],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        ),
    ]


setup(
    cmdclass={'build_ext': BuildExt},
    ext_modules=get_ext_modules(),
)
