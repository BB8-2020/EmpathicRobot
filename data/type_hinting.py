"""Define all NamedTuple for type hinting."""
from typing import NamedTuple

import numpy as np


class DataSplit(NamedTuple):
    """A typing for train/test/val splits."""

    # Train split
    x_train: np.ndarray
    y_train: np.ndarray

    # Val split
    x_val: np.ndarray
    y_val: np.ndarray

    # Test split
    x_test: np.ndarray
    y_test: np.ndarray
