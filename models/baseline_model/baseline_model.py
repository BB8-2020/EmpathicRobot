"""Baseline model using tensorflow. Consists of 3 layers and should only recognize if someone is happy or not."""
from tensorflow.keras import layers, utils
import json
import numpy as np
from typing import Tuple
import tensorflow as tf
from tensorflow import keras


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


def create_datasets(frame: dict, feature: str, target: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create and reshape the datasets for the model. It can be used to create testsets or trainsets.

    Parameters
    ----------
        frame: dict
            Json file that has been converted to a dict using read_data().

        feature: str
            string used to get the correct feature for the dataset from the frame.

        target: str
            string used to get the correct target for the dataset from the frame.

    Return
    ------
        x_feature
            set which contains the features (in this case pictures) for the model.
        y_target
            set which contains the 2 possible targets.
    """
    feature_lst = list(frame[feature].values())

    x_feature = np.array(feature_lst).astype("float32")
    # an image is 48x48 pixels
    x_feature = x_feature.reshape(x_feature.shape[0], 48, 48, 1)
    x_feature /= 255
    target_lst = np.array(list(frame[target].values()))
    # 2 categories: happy and not happy
    y_target = utils.to_categorical(target_lst, 2)
    return x_feature, y_target


def create_model() -> keras.Sequential:
    """
    Create an sequential model that consists of 2 conv layers and 1 dense layer.

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
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(2)])

    return model


def train_model(model: keras.Sequential, frame: dict, batch_size: int, epochs: int, vs: float, save: bool = True) \
        -> keras.callbacks:
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

        save: bool
            choose to save the model as .pd.

    Return
    ------
        history
            The callback where all the training results are saved in. This is used for plotting the training results.
    """
    x_train, y_train = create_datasets(frame, 'formatted_pixels', 'happy')
    history = keras.callbacks.History()
    if save:
        checkpoint_path = "training/cp.ckpt"

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=vs, callbacks=[history, cp_callback])
    else:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=vs, callbacks=[history])

    return history


def compile_model(model: keras.Sequential):
    """
    Compile the model using Adam and CategoricalCrossentropy.

    Parameters
    ----------
        model: tensorflow keras model
                the tensorflow keras model that has been made in create_model().
    """
    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"])


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
    x_test, y_test = create_datasets(frame, 'formatted_pixels', 'happy')
    results = model.evaluate(x_test, y_test, batch_size=batch_size)

    return results
