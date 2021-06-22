"""Tests for the data processing functions of the FerPlus dataset."""
import _pickle as cPickle
import bz2

import numpy as np
import pandas as pd
import pytest

from data.ferPlus.ferPlus_functions import preprocess_data, clean_data_and_normalize, balance_emotions, \
    process_affectnet_data, shuffle_arrays
from data.general_defenitions import split_data


@pytest.fixture
def read_data():
    """Read the FerPlus image set and label set."""
    data = pd.read_csv('tests/dataprocessing/fer2013_sample.csv')
    labels = pd.read_csv('tests/dataprocessing/fer2013new_sample.csv')
    return data, labels


@pytest.fixture
def preprocess_clean(read_data):
    """Preprocess the incoming dataframes and clean/normalize these arrays."""
    x, y = preprocess_data(read_data[0], read_data[1])
    x, y = clean_data_and_normalize(x, y)
    return x, y


@pytest.fixture
def read_extra_affect():
    """Read extra data coming from AffectNet for FerPlus."""
    comp_data = bz2.BZ2File('tests/dataprocessing/fear_disgust_sample.pbz2', 'rb')
    extra_x_train, extra_y_train = cPickle.load(comp_data)
    return extra_x_train, extra_y_train


def test_preprocess_data(read_data):
    """Test the preprocess functions by checking if the images and emotions are the right shape."""
    x, y = preprocess_data(read_data[0], read_data[1])

    size = x[0].shape
    expected_size = (48, 48, 1)
    assert size == expected_size


def test_clean_data_and_normalize():
    """Test the clean and normalize function."""
    x = np.array([255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0])
    y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    x, y = clean_data_and_normalize(x, y)

    assert (int(x[0][0]), y[0].shape) == (1, (7,))


def test_balance_emotions(preprocess_clean):
    """Test to check if the good amount of the emotion as been removed."""
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(preprocess_clean[0], preprocess_clean[1])

    amount_left = 2
    emotion = 'neutral'
    x_train, y_train = balance_emotions(x_train, y_train, emotion, amount_left)

    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
    count_emotions = []
    for i in range(len(x_train)):
        x = y_train[i]
        count_emotions.append(emotions[np.argmax(x)])

    assert count_emotions.count(emotion) == amount_left


def test_process_affectnet_data(preprocess_clean, read_extra_affect):
    """Test if the incoming AffectNet dataset gets formatted and reshaped the good way."""
    extra_x_train, extra_y_train = read_extra_affect[0], read_extra_affect[1]
    x, y = preprocess_clean[0], preprocess_clean[1]

    new_x, new_y = process_affectnet_data(x, y, extra_x_train, extra_y_train)

    assert len(new_x) == (len(x) + len(extra_x_train)) and len(new_y) == (len(y) + len(extra_y_train))


def test_shuffle_arrays(preprocess_clean, read_extra_affect):
    """Test if function shuffles the good way."""
    extra_x_train, extra_y_train = read_extra_affect[0], read_extra_affect[1]

    x, y = process_affectnet_data(preprocess_clean[0], preprocess_clean[1], extra_x_train, extra_y_train)
    shuffled_x, shuffled_y = shuffle_arrays(x, y)

    assert (shuffled_x, shuffled_y) != x, y and (len(shuffled_x), len(shuffled_y)) != (len(x), len(y))
