"""First version of the classification model using VGG16 layout."""
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
from typing import Tuple, Any
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


def read_data(path: str) -> dict:
    """
    Read an pickle file from a given path.

    Parameters
    ----------
        path: str
            Path to the correct pickle file.

    Raises
    ------
        OSError
            When the wrong path is given and it can't find the file.

    Return
    ------
        frame
            pickle data that has been converted to an dictionary.
    """
    try:
        data = open(str(path), 'rb')
        frame = pickle.load(data)
        return frame
    except OSError:
        print(f"File in this {path} does not exist")


def create_datasets(frame: dict, feature: str, target: str) -> Tuple[Any, DataFrame]:
    """
    Create and reshape the datasets for the model. It can be used to create testsets or trainsets.

    Parameters
    ----------
        frame: dict
            Pickle file that has been converted to a dict using read_data().

        feature: str
            string used to get the correct feature for the dataset from the frame.

        target: str
            string used to get the correct target for the dataset from the frame.

    Return
    ------
        x_feature
            set which contains the features (in this case pictures) for the model.
        y_target
            set which contains the 7 possible targets.
    """

    feature_lst = list(frame[feature])
    x_feature = np.array(feature_lst).astype("float32")
    # an image is 48x48 pixels
    x_feature = x_feature.reshape(x_feature.shape[0], 128, 128, 3)
    x_feature /= 255
    x_feature = tf.image.grayscale_to_rgb(
        tf.convert_to_tensor(x_feature),
        name=None)
    y_target = pd.get_dummies(frame[target])
    return x_feature, y_target


def create_model() -> keras.Sequential:
    """
    Create an sequential model.

    Return
    ------
        model
            tensorflow keras model that has been build as seen below.
    """
    model = keras.Sequential()
    model.add(Conv2D(input_shape=(128, 128, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", strides=(1,1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=1000, activation='relu'))
    model.add(Dense(units=7, activation="softmax"))

    return model


def train_model(model: keras.Sequential, frame: dict, batch_size: int, epochs: int, vs: float) -> keras.callbacks:
    """
    Train the model using the trainsets created in create_dataset().

    Parameters
    ----------
        model: tensorflow keras model
            the tensorflow keras model that has been made in create_model().

        frame: dict
            dictionary that is used to create the trainsets in create_datasets().

        batch_size: int
            int that determines the batch_size that is used for training the model.

        epochs: int
            int that determines the amount of epochs which is also used for training the model.

        vs: float
            float that determines the validation_split.

    Return
    ------
        history
            The callback where all the training results are saved in. This is used for plotting the training results.
    """
    history = keras.callbacks.History()

    x_train, y_train = create_datasets(frame, 'formatted_pixels', 'emotion')
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=vs, callbacks=[history])

    return history


def evaluate_model(model: keras.Sequential, frame: dict, batch_size: int) -> list:
    """
    Test the model using testsets created in create_datasets.

    Parameters
    ----------
        model: tensorflow keras model
            the tensorflow keras model that has been made in create_model().

        frame: dict
            dictionary that is used to create the testsets in create_datasets().

        batch_size: int
            int that determines the batch_size that is used for evaluating the model.

    Return
    ------
        results
            list of the results of the model after testing it with te testsets.
    """
    x_test, y_test = create_datasets(frame, 'formatted_pixels', 'emotion')
    results = model.evaluate(x_test, y_test, batch_size=batch_size)

    return results
