"""Tests for the general data processing functions."""
import os
import bz2
import math
import pandas as pd
import _pickle as cPickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data.ferPlus import ferPlus_functions as fer
from data.affectNet import affectNet_functions as affect
from data.general_defenitions import split_data, data_augmentation, show_images, comp_pickle_save

data = pd.read_csv('data/datasets/fer2013.csv')
labels = pd.read_csv('data/datasets/fer2013new.csv')

X, y = fer.preprocess_data(data, labels)
X, y = fer.clean_data_and_normalize(X, y)


def test_split_data():
    """
    Test the split data function if data is devided in correct pieces.

    Assert
    ------
        If all variables have the length they are supposed to get.
    """
    tot_len = len(X)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, y)

    # The test and val variables are supposed to be 10% of the entire length, 20% together.
    length_test = math.ceil(0.1 * tot_len)
    length_val = math.ceil(0.1 * tot_len)
    length_train = tot_len - (length_test + length_val)

    assert (len(x_train), len(x_test), len(x_val)) == (length_train, length_test, length_val)


def test_data_augmentation():
    """
    Test the split data function if data is devided in correct pieces.

    Assert
    ------
        Check if a correct Datagen is created.
    """
    data = bz2.BZ2File('data/datasets/affectNet_val_comp', 'rb')
    df = cPickle.load(data)

    X, y = affect.preprocess_data(df)
    X, y = affect.clean_data_and_normalize(X, y)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, y)
    datagen = data_augmentation(x_train)

    assert type(datagen) == ImageDataGenerator


def test_show_images():
    """
    Test the split data function if data is devided in correct pieces.

    Assert
    ------
        Check if the correct figsizes are plotted in the function.
    """
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, y)
    figsize = show_images(x_train, y_train)

    expected = (10.0, 7.0)
    assert (figsize[0], figsize[1]) == expected

def test_comp_pickle_save():
    """
    Test the save to a compressed pickle file.

    Assert
    ------
        Check if the path to the saved bz2 pickle file exists.
    """
    existing = False

    data = ['test', 'list']
    filename = 'test_comp_pkl_save'
    comp_pickle_save(data, filename)

    if os.path.isfile(filename):
        existing = True
        os.remove(filename)

    assert existing is True
