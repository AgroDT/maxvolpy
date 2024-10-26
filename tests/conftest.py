import numpy as np


def pytest_sessionstart(session) -> None:
    np.random.seed(100)
