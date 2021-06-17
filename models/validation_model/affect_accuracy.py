"""Let our detector guess the emotion of a given amount of affectnet photo's. Calculate the average amount of correct
guesses."""
from typing import List
import pandas as pd
import numpy as np
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
    for row in range(start_row, end_row):
        df.at[row, 'Photo'] = data[0][row]
        df.at[row, 'Correct_emotion'] = data[1][row]
    return df


def aff_accuracy(start_row: int, end_row: int, data: List) -> float:
    """
    Take an amount of photo's from a dataset. And calculate how accurate the emotion detector is with detecting the
    emotions depicted by the people in the photo's. Do this by calculating the percentage of photo's the emotion
    detector got right.

    Parameters
    ----------
        start_row: int
            At which row we want to start taking photo's from our data.

        end_row: int
            After which row we want to stop taking photo's from our data.

        data: List
            In our data you have our photo's and the corresponding emotion.

    Return
    ------
        accuracy: float
            A percentage of how many photo's our detector correctly guessed.
    """
    df = pd.DataFrame(columns=['Photo', 'Correct_emotion'])

    df = update_df(start_row, end_row, data, df)

    # The emotion names from our dataset and the detector aren't always the same.
    # Here we will make those names the same.
    df = df.replace(to_replace=['Happiness', 'Sadness', 'Anger', 'Neutral', 'Fear', 'Surprise', 'Disgust'],
                    value=['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise', 'disgust'])

    # Get the right data in the right variables.
    photos = np.array(df['Photo'].values.tolist()) * 255
    emotions = df['Correct_emotion'].values.tolist()

    # Calculate the accuracy of the trained model with our photo's.
    accuracy = calculate_accuracy(photos, emotions)
    return accuracy
