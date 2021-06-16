"""
We want to know the accuracy of our online found model. To do that, we will test multiple photo's from our
affectnet dataset on the model. The functions to make this possible can be found in this file.
"""

from typing import List
import pandas as pd
from models.validation_model.detector_accuracy import calculate_accuracy


def update_df(start_row: int, end_row: int, data: List, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add photo's to our DataFrame.

    Parameters
    ----------
        start_row: int
            At which row we want to start taking photo's from our data.

        end_row: int
            After which row we want to stop taking photo's from our data.

        data: List
            In our data you have our photo's and the corresponding emotion.

        df: pandas.core.frame.DataFrame
            Our DataFrame.

    Return
    ------
        df: pandas.core.frame.DataFrame
            Our updated DataFrame.
    """
    for row in range(start_row, start_row + end_row):
        df.at[row, 'image'] = data[0][row]
        df.at[row, 'emotion'] = data[1][row]
    return df


def aff_accuracy(start_row: int, end_row: int, data: List, df: pd.DataFrame) -> float:
    """
    We want to take an amount of photo's from our data. Then we also want to know how high the accuracy from the
    detector is.

    Parameters
    ----------
        start_row: int
            At which row we want to start taking photo's from our data.

        end_row: int
            After which row we want to stop taking photo's from our data.

        data: List
            In our data you have our photo's and the corresponding emotion.

        df: pandas.core.frame.DataFrame
            Our DataFrame.

    Return
    ------
        accuracy: float
            A percentage of how many photo's our detector correctly guessed.
    """

    df = update_df(start_row, end_row, data, df)

    # Give the columns a different name.
    df = df.rename(columns={"image": "Photo", 'emotion': 'Correct_emotion'})

    # The emotion names from our dataset and the detector aren't always the same.
    # Here we will make those names the same.
    df = df.replace(to_replace=['Happiness', 'Sadness', 'Anger', 'Neutral', 'Fear', 'Surprise', 'Disgust'],
                    value=['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise', 'disgust'])

    # Get the right data in the right variables.
    photos = df['Photo'].values.tolist()
    emotions = df['Correct_emotion'].values.tolist()

    # Calculate the accuracy of the trained model with our photo's.
    accuracy = calculate_accuracy(photos, emotions)
    return accuracy
