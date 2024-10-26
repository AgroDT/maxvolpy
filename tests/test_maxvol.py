import numpy as np
import pytest

from maxvolpy import rect_maxvol, maxvol


@pytest.fixture(name='arr', scope='session')
def arr_fixture() -> None:
    return np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]


@pytest.mark.parametrize('tol,expected', [(1.0, 1.0000), (1.5, 1.49193), (2.0, 1.91954)])
def test_rect_maxvol(arr: np.ndarray, tol: float, expected: float) -> None:
    piv, c = rect_maxvol(arr, tol)
    assert np.allclose(arr, c.dot(arr[piv]))
    max_ = max(np.linalg.norm(c[i], 2) for i in range(1000))
    assert round(max_, 5) == expected


@pytest.mark.parametrize('tol,expected', [(1.0, 1.0000), (1.05, 1.04641), (1.1, 1.07854)])
def test_maxvol(arr: np.ndarray, tol: float, expected: float) -> None:
    piv, c = maxvol(arr, tol)
    assert np.allclose(arr, c.dot(arr[piv]))
    max_ = np.abs(c).max().round(5)
    assert max_ == expected
