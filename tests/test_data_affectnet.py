"""Tests for the data processing functions of the AffectNet dataset."""
import _pickle as cPickle
import bz2
import os

import numpy as np
import pandas as pd

from data.affectNet.affectNet_functions import preprocess_data, clean_data_and_normalize, get_latest_index, \
    convert_to_dataframe


def test_get_latest_index():
    """Test the latest index function where you can the index of the last image in the current folder."""
    os.chdir(os.getcwd() + '/tests/dataprocessing')
    output = get_latest_index()
    expected = int
    assert type(output) == expected


def test_convert_to_dataframe():
    """Test the convert to dataframe function to see if a correct dataframe is created."""
    emotions = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger',
                7: 'Contempt', 8: 'None', 9: 'Uncertain', 10: 'No-Face'}

    cap = 1
    path = 'train_set'
    latest_img = 2

    df = pd.DataFrame(columns=['formatted_pixels', 'target'])

    convert_to_dataframe(latest_img, df, emotions, path, cap)

    assert len(df['target']) == 2


def test_preprocess_data():
    """Test the preprocess functions by checking if the images and emotions are the right shape."""
    data = bz2.BZ2File('affectNet_sample.pbz2', 'rb')
    df = cPickle.load(data)

    X, y = preprocess_data(df)
    size = np.array(X[0]).shape
    expected_size = (48, 48, 3)
    assert size == expected_size


def test_clean_data_and_normalize():
    """Test the clean and normalize function."""
    X = np.array([255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0])
    y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    X, y = clean_data_and_normalize(X, y)
    os.chdir(os.getcwd() + '/../../')

    assert (int(X[0]), y.shape) == (1, (10, 10))
