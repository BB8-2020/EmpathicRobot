"""General definitions that are used for cleaning, preparing and processing the FerPlus and AffectNet dataset."""
import _pickle as cPickle
import bz2
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.data.type_hinting import DataSplit


def split_data(x: np.ndarray, y: np.ndarray or pd.Dataframe) -> DataSplit:
    """
    Split the incoming data into train, test and validation sets.

    Parameters
    ----------
        x: np.ndarray
            All features (images)
        y: np.ndarray or pd.Dataframe
            All targets (emotions), FerPlus: array and AffectNet: dataframe
    Return
    ------
        x_train:
            The train features (images)
        y_train:
            The train targets (emotions)
        x_val:
            The validation features (images)
        y_val:
            The validation targets (emotions)
        x_test:
            The test features (images)
        y_test
            The test targets (emotions)
    """
    test_size = ceil(len(x) * 0.1)

    # Split Data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=test_size, random_state=42
    )
    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        x_train, y_train, test_size=test_size, random_state=42
    )
    return DataSplit(x_train, y_train, x_val, y_val, x_test, y_test)


def data_augmentation(x_train: np.ndarray) -> ImageDataGenerator:
    """
    Augment the images by rotating, flipping and shifting.

    Parameters
    ----------
        x_train: np.ndarray
            The processed train features (images)
    Return
    ------
        datagen: np.ndarray
            The processed train images augmented
    """
    shift = 0.1
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        height_shift_range=shift,
        width_shift_range=shift,
    )
    datagen.fit(x_train)
    return datagen


def show_images(x_train: np.ndarray, y_train: np.ndarray, datagen: ImageDataGenerator = None, amount: int = 25) -> list:
    """
    Show images with a check for augmented images and check if the emotion is defined to show.

    Parameters
    ----------
        x_train: np.ndarray
            The train features (images)
        y_train: np.ndarray
            The train targets (emotions)
        datagen: np.ndarray = None
            The augmented images, can be None
        amount: int = 25
            The amount of images you want to get plotted
    Return
    ------
        fig.get_size_inches(): list
            The size of the made figs to test this function
    """
    it = datagen.flow(x_train, y_train, batch_size=1) if datagen else None

    fig = plt.figure(figsize=(10, 7))
    for i in range(amount):
        plt.subplot(5, 5, i + 1)
        plt.imshow(np.squeeze(it.next()[0][0]), cmap="gray") if datagen else plt.imshow(np.squeeze(x_train[i]),
                                                                                        cmap="gray")

    plt.show()
    return fig.get_size_inches()


def comp_pickle_save(data: list, filename: str) -> None:
    """
    Dump the incoming data in a compressed pickle file,
    this is one of the lightest ways to save data with and keep easy access.

    Parameters
    ----------
        data: list
            All data wanted to be saved in a compressed pickle file
        filename: str
            Wanted filename for the file
    """
    with bz2.BZ2File(filename, "w") as f:
        cPickle.dump(data, f)
