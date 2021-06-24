"""Tests for the data processing functions of the AffectNet dataset."""
import _pickle as cPickle
import bz2

import numpy as np
import pandas as pd

from src.data import preprocess_data, clean_data_and_normalize, convert_to_dataframe


def test_convert_to_dataframe():
    """Test the convert to dataframe function to see if a correct dataframe is created."""
    emotions = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger',
                7: 'Contempt', 8: 'None', 9: 'Uncertain', 10: 'No-Face'}

    cap = 1
    path = 'tests/dataprocessing/train_set'

    df = pd.DataFrame(columns=['formatted_pixels', 'target'])

    convert_to_dataframe(df, emotions, path, cap)

    assert len(df['target']) == 2


def test_preprocess_data():
    """Test the preprocess functions by checking if the images and emotions are the right shape."""
    data = bz2.BZ2File('tests/dataprocessing/affectNet_sample.pbz2', 'rb')
    df = cPickle.load(data)

    x, y = preprocess_data(df)
    size = np.array(x[0]).shape
    expected_size = (48, 48, 3)
    assert size == expected_size


def test_clean_data_and_normalize():
    """Test the clean and normalize function."""
    x = np.array([255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0])
    y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    x, y = clean_data_and_normalize(x, y)

    assert (int(x[0]), y.shape) == (1, (10, 10))
