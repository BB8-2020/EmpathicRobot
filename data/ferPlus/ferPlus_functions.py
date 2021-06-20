"""Functions for processing, cleaning and normalizing the FerPlus dataset."""
from typing import Tuple

import numpy as np
import pandas as pd


def preprocess_data(data: pd.DataFrame, labels: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define the needed size of the image and turn the needed columns into numpy arrays.

    Parameters
    ----------
        data: pd.DataFrame
            First dataframe containing the features (images)
        labels: pd.DataFrame
            Second dataframe containing the targets (emotions)
    Return
    ------
        x: np.ndarray
            All features (images)
        y: np.ndarray
            All targets (emotions)
    """
    orig_class_names = [
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
        "unknown",
        "NF",
    ]

    n_samples = len(data)
    w = 48
    h = 48
    y = np.array(labels[orig_class_names])
    x = np.zeros((n_samples, w, h, 1))
    for i in range(n_samples):
        x[i] = np.fromstring(data["pixels"][i], dtype=int, sep=" ").reshape((h, w, 1))
    return x, y


def clean_data_and_normalize(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove the unnecessary columns and normalize all data.

    Parameters
    ----------
        x: np.ndarray
            All features (images)
        y: np.ndarray
            All targets (emotions)
    Return
    ------
        x: np.ndarray
            All  cleaned and normalized features (images)
        y: np.ndarray
            All  cleaned and normalized targets (emotions)
    """
    orig_class_names = [
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
        "unknown",
        "NF",
    ]

    # Using mask to remove unknown or NF images
    y_mask = y.argmax(axis=-1)
    mask = y_mask < orig_class_names.index("unknown")
    x, y = x[mask], y[mask]

    # Convert to probabilities between 0 and 1
    y = y[:, :-2] * 0.1

    # Add contempt to neutral and remove it
    y[:, 0] += y[:, 7]
    y = y[:, :7]

    # Normalize image vectors
    x /= 255.0

    return x, y


def balance_emotions(x_train: np.ndarray, y_train: np.ndarray, emotion: str, amount_left: int):
    """
    Remove the excess data where you want to balance the data.

    Parameters
    ----------
        x_train: np.ndarray
            All features (images)
        y_train: np.ndarray
            All targets (emotions)
        emotions: string
            Emotions you want to balance
        emount_left: int
            Amount of this emotion you want to have left
    Return
    ------
        x_train: np.ndarray
            All features with the amount_left of the images
        y_train: np.ndarray
            All targets with the amount_left of the emotions
    """
    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
    lst_emotions = []
    for i in range(len(x_train)):
        x = y_train[i]
        lst_emotions.append(emotions[np.argmax(x)])

    x_train, y_train = list(x_train), list(y_train)
    count = 0
    treshold_amount_of_emotion = lst_emotions.count(emotion) - amount_left
    indexes = []

    for index in range(len(x_train)):
        if count >= treshold_amount_of_emotion:
            break
        if emotions[np.argmax(y_train[index])] == emotion:
            indexes.append(index)
            count += 1

    indexes.reverse()

    for i in indexes:
        x_train.pop(i)
        y_train.pop(i)

    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train
