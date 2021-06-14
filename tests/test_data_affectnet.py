"""Tests for the data processing functions of the AffectNet dataset."""
import bz2
import numpy as np
import pandas as pd
import _pickle as cPickle
import os

from data.affectNet.affectNet_functions import preprocess_data, clean_data_and_normalize, get_latest_index, \
    convert_to_dataframe


def test_get_latest_index():
    """
    Test the latest index function where you can the index of the last image in the current folder.

    Assert
    ------
        We check if the returned index is an integer.

    """
    os.chdir(os.getcwd() + '/../test_data_lead')
    output = get_latest_index()
    expected = int
    assert type(output) == expected


def test_convert_to_dataframe():
    """
    Test the convert to dataframe function to see if a correct dataframe is created.

    Assert
    ------
        Check if the data frame is correctly filled with the values.
    """
    emotions = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger',
                7: 'Contempt', 8: 'None', 9: 'Uncertain', 10: 'No-Face'}

    cap = 2
    path = 'train_set'
    latest_img = 5

    df = pd.DataFrame(columns=['formatted_pixels', 'target'])

    convert_to_dataframe(latest_img, df, emotions, path, cap)

    assert len(df['target']) == 3


def test_preprocess_data():
    """
    Test the preprocess functions by checking if the images and emotions are the right shape.

    Assert
    ------
        Check if the outcoming size of images equals the wanted shape.
    """
    data = bz2.BZ2File('../EmpathicRobot/data/datasets/affectNet_val_comp', 'rb')
    df = cPickle.load(data)

    X, y = preprocess_data(df)
    size = np.array(X[0]).shape
    expected_size = (48, 48, 3)
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
    os.chdir(os.getcwd() + '/../EmpathicRobot')

    assert (int(X[0]), y.shape) == (1, (10, 10))
