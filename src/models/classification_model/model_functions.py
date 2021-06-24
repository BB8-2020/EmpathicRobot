"""Functions that are needed to read data, train model en evaluate it."""
import _pickle as cPickle
from bz2 import BZ2File
from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras


def read_data(path: str, datagen: bool = False) -> Tuple:
    """
    Read the data out the compressed pickle file and split the data into train, test and validation set.

    Parameters
    ----------
        path: str
            the path to the compressed file
        datagen: bool
            check if the data argument
    Return
    ------
        Tuple of the data sets
    """
    if datagen:
        datagen, x_train, y_train, x_val, y_val, x_test, y_test = cPickle.load(BZ2File(str(path), 'rb'))
        return datagen, x_train, y_train, x_val, y_val, x_test, y_test

    else:
        x_train, y_train, x_val, y_val, x_test, y_test = cPickle.load(BZ2File(str(path), 'rb'))
        return x_train, y_train, x_val, y_val, x_test, y_test


def fit_model(model: keras.Sequential, batch_size: int = 64, epochs: int = 100, dategen: bool = False, *data) ->\
        keras.callbacks:
    """
    Fit model using the standard keras fit function.

    Parameters
    ----------
        model: keras.Sequential
            the model that would be trained
        batch_size: int
            determines the batch_size that is used for training the model
        epochs: int
            determines the amount of epochs which is also used for training the model
        dategen: bool
            check if the train for the argument data
        data
            the datasets that has been split
    Return
    ------
     history: keras.callbacks
            where all the training results are saved in. This is used for ploty catching the training results.

    """

    if dategen:
        datagen, x_train, y_train, x_val, y_val, x_test = data
        history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            steps_per_epoch=len(x_train) // batch_size,
                            validation_data=(x_val, y_val), verbose=2)
    else:
        x_train, y_train, x_val, y_val, x_test = data

        history = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs,
                            steps_per_epoch=len(x_train) // batch_size,
                            validation_data=(x_val, y_val), verbose=2)
    return history


def compile_model(model: keras.Sequential) -> None:
    """
    Compile the model using Adam and binary cross entropy.

    Parameters
    ----------
        model: keras.Sequential
            the model that should be compiled
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


def evaluate_model(model: keras.Sequential, x_test: List[float], y_test: List[float], batch_size: int = 64) \
        -> Tuple[float, float]:
    """
    Evaluate model using the test set.

    Parameters
    ----------
        model: keras.Sequential
            the model that should be tested
        x_test: list
            the features of the test set
        y_test: list
            the targets of the test set
        batch_size: int
            determines the batch_size that is used for test the model
    Return
    ------
        tuple of the test acc and the train acc of the model.
    """
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    return test_loss, test_acc
