import os

import cv2
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


def get_latest_index():
    """
    Get index of the last image in folder.

    Return
    ------
        latest_img: int
            Index of last image in folder
    """
    latest_img = 0
    arr = os.listdir("train_set/images")

    for i in arr:
        index = i.split(".")[0]
        if int(index) > latest_img:
            latest_img = int(index)

    return latest_img


def convert_to_dataframe(latest_img: int, df: pd.DataFrame, emotions: dict, path: str, cap: int = 28000):
    """
    Loop through all images, except for datasets larger than the wanted size.
    Load these images with labels into a pandas dataframe.

    Parameters
    ----------
        latest_img: int
            Index of last image in folder
        df: pd.DataFrame
            Dataframe that needs to be filled with features and targets
        emotions: list
            All emotions that can occur in the dataframe
        path: str
            Path where images and emotions are settled
        cap: int = 28000
            max size of a dataframe we create
    Return
    ------
        X: np.ndarray
            All  cleaned and normalized features (images)
        y: np.ndarray
            All  cleaned and normalized targets (emotions)
    """
    frame_index = 0

    # Because of the heavy weight of this dataset we decided to max it at 28 images.
    if latest_img > cap:
        latest_img = cap

    for i in range(latest_img + 1):
        try:
            emotion = np.load(path + "/annotations/" + str(i) + "_exp.npy")
            img = mpimg.imread(path + "/images/" + str(i) + ".jpg")

            df.at[frame_index, "formatted_pixels"] = img
            df.at[frame_index, "target"] = emotions[int(emotion)]

            frame_index += 1

        # File not found is part of an IOE error.
        except OSError:
            print(f"Image with index {i} was not found.")


def preprocess_data(data: pd.DataFrame):
    """
    Resize the images based on what size de model is trained on. Turn the needed columns into numpy arrays.

    Parameters
    ----------
        data: pd.DataFrame
            Dataframe containing all data
    Return
    ------
        X: np.ndarray
            Array containing all resized images
        y: np.ndarray
            Array containing all emotions
    """
    n_samples = len(data)
    width = 48
    height = 48

    y = np.array(data["target"])
    X = np.zeros((n_samples, width, height, 3))

    for i in range(n_samples):
        X[i] = cv2.resize(
            data["formatted_pixels"].iloc[i],
            dsize=(width, height),
            interpolation=cv2.INTER_CUBIC,
        )

    return X, y


def clean_data_and_normalize(X: np.ndarray, y: np.ndarray):
    """
    Normalize the data.

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
    # Normalize image vectors
    y = pd.get_dummies(y)
    X /= 255.0

    return X, y
