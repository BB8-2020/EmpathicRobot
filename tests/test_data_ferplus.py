"""Tests for the data processing functions of the FerPlus dataset."""
import random
import pytest
import numpy as np
import pandas as pd
from data.ferPlus.ferPlus_functions import preprocess_data, clean_data_and_normalize


def test_preprocess_data():
    """
    Test the preprocess functions by checking if the images and emotions are the right shape.

    Assert
    ------
        Check if the outcoming size of images equals the wanted shape.
    """
    pixelstring = [random.randrange(0, 248) for i in range(2304)]
    emotion = 'happiness'

    data = {'pixels': [pixelstring],
            'emotion': [emotion]
            }

    df = pd.DataFrame(data, columns=['pixels', 'emotion'])

    preprocess_data(df, df)
    size = np.array(df['pixels'][0]).shape
    expected_size = (48, 48)
    assert size == expected_size

def test_clean_data_and_normalize():
    """
    Test the clean and normalize function

    Assert
    ------
        Check if the target values are normalized and dummied.
    """
    X = np.array([255, 255, 255, 255, 255, 255, 255])
    y = np.array([0, 1, 2, 3, 4, 5, 6])

    X, y = clean_data_and_normalize(X, y)

    assert (int(X[0]), y[0].shape) == (1, (7, 7))
