"""Functions for converting, cleaning, preparing and processing the AffectNet dataset."""
import os
from typing import Tuple

import cv2
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


def convert_to_dataframe(df: pd.DataFrame, emotions: dict, path: str, cap: int = 28000) -> None:
    """
    Loop through all images, except for datasets larger than the wanted size.
    Load these images with labels into a pandas dataframe.

    Parameters
    ----------
        df: pd.DataFrame
            Dataframe that needs to be filled with features and targets
        emotions: list
            All emotions that can occur in the dataframe
        path: str
            Path where images and emotions are settled
        cap: int = 28000
            max size of a dataframe we create
    """
    # Get the name of last image in folder
    latest_img = int(os.listdir(f"{path}/images")[-1].split(".")[0])

    frame_index = 0

    # Because of the heavy weight of this dataset we decided to max it at 28k images.
    if latest_img > cap:
        latest_img = cap

    for i in range(latest_img + 1):
        try:
            emotion = np.load(f"{path}/annotations/{i}_exp.npy")
            img = mpimg.imread(f"{path}/images/{i}.jpg")

            df.at[frame_index, "formatted_pixels"] = img
            df.at[frame_index, "target"] = emotions[int(emotion)]

            frame_index += 1

        # File not found is part of an IOE error.
        except OSError:
            print(f"Image with index {i} was not found.")


def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize the images based on what size de model is trained on. Turn the needed columns into numpy arrays.

    Parameters
    ----------
        data: pd.DataFrame
            Dataframe containing all data
    Return
    ------
        x: np.ndarray
            Array containing all resized images
        y: np.ndarray
            Array containing all emotions
    """
    n_samples = len(data)
    width = 48
    height = 48

    y = np.array(data["target"])
    x = np.zeros((n_samples, width, height, 3))

    for i in range(n_samples):
        x[i] = cv2.resize(
            data["formatted_pixels"].iloc[i],
            dsize=(width, height),
            interpolation=cv2.INTER_CUBIC,
        )

    return x, y


def clean_data_and_normalize(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Normalize the data.

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
    # Normalize image vectors
    y = pd.get_dummies(y)
    x /= 255.0

    return x, y
