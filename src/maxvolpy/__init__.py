"""
Module `maxvolpy` is designed for constructing different low-rank
skeleton and cross approximations.

Right now, cross approximation techniques are not implemented yet, but
all kinds of algorithms of finding good submatrices to build skeleton
approximations are presented in `maxvol` submodule. What does good
submatrix mean is noted in documentation for `maxvol` submodule.
"""

from __future__ import annotations

__all__ = ['__version__', 'rect_maxvol', 'maxvol']

from .__version__ import __version__

try:
    from ._maxvol import rect_maxvol, maxvol
except ImportError:
    import warnings

    warnings.warn(
        'fast C maxvol functions are not compiled,'
        ' continue with python maxvol functions',
        RuntimeWarning,
    )
    from .maxvol import rect_maxvol, maxvol
