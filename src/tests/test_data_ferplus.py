"""Tests for the data processing functions of the FerPlus dataset."""
import numpy as np
import pandas as pd

from src.data import preprocess_data, clean_data_and_normalize


def test_preprocess_data():
    """Test the preprocess functions by checking if the images and emotions are the right shape."""
    data = pd.read_csv('tests/dataprocessing/fer2013_sample.csv')
    labels = pd.read_csv('tests/dataprocessing/fer2013new_sample.csv')

    X, y = preprocess_data(data, labels)

    size = X[0].shape
    expected_size = (48, 48, 1)
    assert size == expected_size


def test_clean_data_and_normalize():
    """Test the clean and normalize function."""
    X = np.array([255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0])
    y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    X, y = clean_data_and_normalize(X, y)

    assert (int(X[0][0]), y[0].shape) == (1, (7,))
