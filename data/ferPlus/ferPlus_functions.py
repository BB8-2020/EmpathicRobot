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
        X: np.ndarray
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
    X = np.zeros((n_samples, w, h, 1))
    for i in range(n_samples):
        X[i] = np.fromstring(data["pixels"][i], dtype=int, sep=" ").reshape((h, w, 1))
    return X, y


def clean_data_and_normalize(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove the unnecessary columns and normalize all data.

    Parameters
    ----------
        X: np.ndarray
            All features (images)
        y: np.ndarray
            All targets (emotions)
    Return
    ------
        X: np.ndarray
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
    X, y = X[mask], y[mask]

    # Convert to probabilities between 0 and 1
    y = y[:, :-2] * 0.1

    # Add contempt to neutral and remove it
    y[:, 0] += y[:, 7]
    y = y[:, :7]

    # Normalize image vectors
    X /= 255.0

    return X, y
