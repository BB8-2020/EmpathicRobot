"""Tests for the general data processing functions."""
import _pickle as cPickle
import bz2
import math
import os

import pandas as pd
import pytest
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.data.affectNet import affectNet_functions as affect
from src.data.ferPlus import ferPlus_functions as fer
from src.data.general_defenitions import split_data, data_augmentation, show_images, comp_pickle_save


@pytest.fixture
def read_process_clean_data():
    """Read, process and clean/normalize the FerPlus dataset."""
    data = pd.read_csv('src/tests/dataprocessing/fer2013_sample.csv')
    labels = pd.read_csv('src/tests/dataprocessing/fer2013new_sample.csv')

    x, y = fer.preprocess_data(data, labels)
    x, y = fer.clean_data_and_normalize(x, y)
    return x, y


def test_split_data(read_process_clean_data):
    """Test the split data function if data is devided in correct pieces."""
    x, y = read_process_clean_data[0], read_process_clean_data[1]

    total_len = len(x)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y)

    # The test and val variables are supposed to be 10% of the entire length, 20% together.
    length_test_train = math.ceil(0.1 * total_len)
    length_train = total_len - (length_test_train * 2)

    assert (len(x_train), len(x_test), len(x_val)) == (length_train, length_test_train, length_test_train)


def test_data_augmentation():
    """Test the split data function if data is devided in correct pieces."""
    data = bz2.BZ2File('src/tests/dataprocessing/affectNet_sample.pbz2', 'rb')
    df = cPickle.load(data)

    x, y = affect.preprocess_data(df)
    x, y = affect.clean_data_and_normalize(x, y)
    x_train = split_data(x, y)[0]
    datagen = data_augmentation(x_train)

    assert type(datagen) == ImageDataGenerator


def test_show_images(read_process_clean_data):
    """Test the split data function if data is devided in correct pieces."""
    x_train, y_train, _, _, _, _ = split_data(read_process_clean_data[0], read_process_clean_data[1])
    figsize = show_images(x_train, y_train, None, 5)

    # This is the expected shape for the figures that are made.
    expected = (10.0, 7.0)
    assert (figsize[0], figsize[1]) == expected


def test_comp_pickle_save():
    """Test the save to a compressed pickle file."""
    data = ['test', 'list']
    filename = 'test_comp_pkl_save'
    comp_pickle_save(data, filename)

    assert os.path.isfile(filename) is True