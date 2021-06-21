"""Tests for the data processing functions of the FerPlus dataset."""
import numpy as np
import pandas as pd

from data.ferPlus.ferPlus_functions import preprocess_data, clean_data_and_normalize, balance_emotions
from data.general_defenitions import split_data


def test_preprocess_data():
    """Test the preprocess functions by checking if the images and emotions are the right shape."""
    data = pd.read_csv('tests/dataprocessing/fer2013_sample.csv')
    labels = pd.read_csv('tests/dataprocessing/fer2013new_sample.csv')

    x, y = preprocess_data(data, labels)

    size = x[0].shape
    expected_size = (48, 48, 1)
    assert size == expected_size


def test_clean_data_and_normalize():
    """Test the clean and normalize function."""
    x = np.array([255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0])
    y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    x, y = clean_data_and_normalize(x, y)

    assert (int(x[0][0]), y[0].shape) == (1, (7,))


def test_balance_emotions():
    """"Test to check if the good amount of the emotion as been removed."""
    data = pd.read_csv('tests/dataprocessing/fer2013_sample.csv')
    labels = pd.read_csv('tests/dataprocessing/fer2013new_sample.csv')

    x, y = preprocess_data(data, labels)
    x, y = clean_data_and_normalize(x, y)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y)

    amount_left = 2
    emotion = 'neutral'
    x_train, y_train = balance_emotions(x_train, y_train, emotion, amount_left)

    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
    count_emotions = []
    for i in range(len(x_train)):
        x = y_train[i]
        count_emotions.append(emotions[np.argmax(x)])

    assert count_emotions.count(emotion) == amount_left
