"""Create the model settings, compile, train and test functions."""
import _pickle as cPickle
from bz2 import BZ2File
from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization


def build_models(input_shape: Tuple[int, int, int] = (48, 48, 1), num_classes: int = 7) -> List[dict]:
    """
    Build the models settings using a list. This is used to train the model at the same time.

    Parameters
    ----------
        input_shape: Tuple
            the input shape of the model
        num_classes: int
            the number of classes for the prediction
    Return
    ------
         models_settings: list
            the models settings
    """
    num_features = 64
    activation_ = 'relu'

    models_settings = [{
        "name": "Version_1",
        "layers": [
                # 1st stage
                Conv2D(num_features, kernel_size=(3, 3), input_shape=input_shape, activation=activation_),
                BatchNormalization(), Dropout(0.5),
                # 2nd stage
                Conv2D(num_features, (3, 3), activation=activation_),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                # 3rd stage
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                # 4th stage
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                # 5th stage
                Conv2D(4 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                # Fully connected neural networks
                Flatten(), Dense(1024, activation=activation_), Dropout(0.2),
                Dense(1024, activation=activation_), Dropout(0.2), Dense(num_classes, activation='softmax')
        ]},

        {
        "name": "Version_2",
        "layers": [
                # 1st stage
                Conv2D(num_features, kernel_size=(3, 3), input_shape=input_shape, activation=activation_),
                BatchNormalization(),
                Conv2D(num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                Dropout(0.5),
                # 2nd stage
                Conv2D(num_features, (3, 3), activation=activation_),
                Conv2D(num_features, (3, 3), activation=activation_),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                # 3rd stage
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                # 4rd stage
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(4 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                # 5th stage
                Conv2D(4 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                # Fully connected neural networks
                Flatten(), Dense(1024, activation=activation_), Dropout(0.2),
                Dense(1024, activation=activation_), Dropout(0.2), Dense(num_classes, activation='softmax')
        ]
    }]
    return models_settings


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
