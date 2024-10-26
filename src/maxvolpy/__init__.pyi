from typing import TypeAlias, TypeVar

import numpy as np

_T = TypeVar('_T', np.dtype[np.float32], np.dtype[np.float64], np.dtype[np.complex64], np.dtype[np.complex128])
_Result: TypeAlias = tuple[np.ndarray[tuple[int], np.dtype[np.int32]], np.ndarray[tuple[int, int], _T]]

def rect_maxvol(
    A: np.ndarray[tuple[int, int], _T],
    tol: float | None = 1.0,
    maxK: int | None = None,
    min_add_K: int | None = None,
    minK: int | None = None,
    start_maxvol_iters: int | None = 10,
    identity_submatrix: bool | None = True,
    top_k_index: int | None = -1,
) -> _Result[_T]:
    """
    Finds good rectangular submatrix.

    Uses greedy iterative maximization of 2-volume to find good
    `K`-by-`r` submatrix in a given `N`-by-`r` matrix `A` of rank `r`.
    Returns good submatrix and least squares coefficients of expansion
    (`N`-by-`K` matrix) of rows of matrix `A` by rows of good submatrix.

    Parameters
    ----------
    A : numpy.ndarray(ndim=2)
        Real or complex matrix of shape `(N, r)`, `N >= r`.
    tol : float, optional
        Upper bound for euclidian norm of coefficients of expansion of
        rows of `A` by rows of good submatrix. Defaults to `1.0`.
    maxK : integer, optional
        Maximum number of rows in good submatrix. Defaults to `N` if
        not set explicitly.
    minK : integer, optional
        Minimum number of rows in good submatrix. Defaults to `r` if
        not set explicitly.
    min_add_K : integer, optional
        Minimum number of rows to add to the square submatrix.
        Resulting good matrix will have minimum of `r+min_add_K` rows.
        Ignored if not set explicitly.
    start_maxvol_iters : integer, optional
        How many iterations of square maxvol (optimization of 1-volume)
        is required to be done before actual rectangular 2-volume
        maximization. Defaults to `10`.
    identity_submatrix : boolean, optional
        Coefficients of expansions are computed as least squares
        solution. If `identity_submatrix` is True, returned matrix of
        coefficients will have submatrix, corresponding to good rows,
        set to identity. Defaults to `True`.
    top_k_index : integer, optional
        Pivot rows for good submatrix will be in range from `0` to
        `(top_k_index-1)`. This restriction is ignored, if `top_k_index`
        is -1. Defaults to `-1`.

    Returns
    -------
    piv : numpy.ndarray(ndim=1, dtype=numpy.int32)
        Rows of matrix `A`, corresponding to submatrix, good in terms
        of 2-volume. Shape is `(K, )`.
    C : numpy.ndarray(ndim=2)
        Matrix of coefficients of expansions of all rows of `A` by good
        rows `piv`. Shape is `(N, K)`.

    Examples
    --------
    >>> import numpy as np
    >>> from maxvolpy.maxvol import rect_maxvol
    >>> np.random.seed(100)
    >>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
    >>> piv, C = rect_maxvol(a, 1.0)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.00000
    >>> piv, C = rect_maxvol(a, 1.5)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.49193
    >>> piv, C = rect_maxvol(a, 2.0)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.91954
    """

def maxvol(
    A: np.ndarray[tuple[int, int], _T],
    tol: float | None = 1.0,
    max_iters: int | None = 100,
    top_k_index: int | None = -1,
) -> _Result[_T]:
    """
    Finds good square submatrix.

    Uses greedy iterative maximization of 1-volume to find good
    `r`-by-`r` submatrix in a given `N`-by-`r` matrix `A` of rank `r`.
    Returns good submatrix and coefficients of expansion
    (`N`-by-`r` matrix) of rows of matrix `A` by rows of good submatrix.

    Parameters
    ----------
    A : numpy.ndarray(ndim=2)
        Real or complex matrix of shape `(N, r)`, `N >= r`.
    tol : float, optional
        Upper bound for infinite norm of coefficients of expansion of
        rows of `A` by rows of good submatrix. Minimum value is 1.
        Default to `1.05`.
    max_iters : integer, optional
        Maximum number of iterations. Each iteration swaps 2 rows.
        Defaults to `100`.
    top_k_index : integer, optional
        Pivot rows for good submatrix will be in range from `0` to
        `(top_k_index-1)`. This restriction is ignored, if `top_k_index`
        is -1. Defaults to `-1`.

    Returns
    -------
    piv : numpy.ndarray(ndim=1, dtype=numpy.int32)
        Rows of matrix `A`, corresponding to submatrix, good in terms
        of 1-volume. Shape is `(r, )`.
    C : numpy.ndarray(ndim=2)
        Matrix of coefficients of expansions of all rows of `A` by good
        rows `piv`. Shape is `(N, r)`.

    Examples
    --------
    >>> import numpy as np
    >>> from maxvolpy.maxvol import maxvol
    >>> np.random.seed(100)
    >>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
    >>> piv, C = maxvol(a, 1.0)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.00000
    >>> piv, C = maxvol(a, 1.05)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.04641
    >>> piv, C = maxvol(a, 1.10)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.07854
    """
