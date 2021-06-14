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
    pixelstring = ''
    for i in range(2304):
        x = random.randrange(0, 248)
        pixelstring += f'{x} '

    data = {'pixels': [pixelstring],
            "happiness": [0],
            "surprise": [0],
            "sadness": [0],
            "anger": [6],
            "disgust": [1],
            "fear": [3],
            "contempt": [0],
            "unknown": [0],
            "NF": [0],
             }

    df = pd.DataFrame(data, columns=['pixels', 'happiness', "neutral", 'surprise', "sadness", "anger", "disgust", "fear", "contempt", "unknown", "NF",])

    X, y = preprocess_data(df, df)
    size = X[0].shape
    expected_size = (48, 48, 1)
    assert size == expected_size


def test_clean_data_and_normalize():
    """
    Test the clean and normalize function

    Assert
    ------
        Check if the target values are normalized and dummied to the right shapes.
    """
    X = np.array([255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0])
    y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    X, y = clean_data_and_normalize(X, y)

    assert (int(X[0][0]), y[0].shape) == (1, (7,))
