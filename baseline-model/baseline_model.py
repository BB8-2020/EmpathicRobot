"""Baseline model using tensorflow. Consists of 3 layers and should only recognize if someone is happy or not."""
from tensorflow import keras
from tensorflow.keras import layers, utils
import json
import numpy as np
from typing import Tuple


def read_data(path: str) -> dict:
    """
    Read an json file from a given path.

    Parameters
    ----------
        path: str
            Path to the correct json file.

    Raises
    ------
        OSError
            When the wrong path is given and it can't find the file.

    Return
    ------
        frame
            json data that has been converted to an dictionary.
    """
    try:
        data = open(str(path))
        frame = json.loads(data.read())
        return frame
    except OSError:
        print(f"File in this {path} does not exist")


def create_trainsets(frame: dict, feature: str, target: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create and reshape the train sets for the model.

    Parameters
    ----------
        frame: dict
            Json file that has been converted to a dict using read_data().

        feature: str
            string used to get the correct feature for the training data from the frame.

        target: str
            string used to get the correct target for the training data from the frame.

    Return
    ------
        x_train
            training set which contains the features (in this case pictures) for the model.
        y_train
            training set which contains the 2 possible targets.
    """
    feature_lst = list(frame[feature].values())

    x_train = np.array(feature_lst).astype("float32")
    # an image is 48x48 pixels
    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train /= 255
    target_lst = np.array(list(frame[target].values()))
    # 2 categories: happy and not happy
    y_train = utils.to_categorical(target_lst, 2)
    return x_train, y_train


def create_model() -> keras.Sequential:
    """
    Create an sequential model that consists of 3 layers.

    Return
    ------
        model
            tensorflow keras model that has been build as seen below.
    """
    model = keras.Sequential([
        keras.Input(shape=(48, 48, 1)),
        layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu', name='conv1'),
        layers.BatchNormalization(axis=-1),
        layers.Conv2D(kernel_size=(3, 3), filters=64, activation='relu', name='conv2'),
        layers.BatchNormalization(axis=-1),
        layers.Conv2D(kernel_size=(3, 3), filters=128, activation='relu', name='conv3'),
        layers.BatchNormalization(axis=-1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(2, activation='softmax')])

    return model


def train_model(model: keras.Sequential, frame: dict, batch_size: int, epochs: int, vs: float) -> None:
    """
    Train the model using the trainsets created in create_trainsets().

    Parameters
    ----------
        model: tensorflow keras model
            the tensorflow keras model that has been made in create_model().

        frame: dict
            dictionary that is used to create the trainsets in create_trainsets().

        batch_size: int
            int that determines the batch_size that is used for training the model.

        epochs: int
            int that determines the amount of epochs which is also used for training the model.

        vs: float
            float that determines the validation_split.
    """
    x_train, y_train = create_trainsets(frame, 'formatted_pixels', 'happy')
    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=vs)
