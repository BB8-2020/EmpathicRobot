"""In this document there are functions that will go through a series of photo's and will try to guess the correct
emotion depicted there."""
import numpy as np
from typing import List
from fer import FER


def recognise_emotion(photo: np.ndarray, correct_emotion: str, detector: FER) -> bool:
    """
    Give emotion detector an image. Return whether or not our detector has guessed the correct emotion depicted by the
    person in the image.

    Parameters
    ----------
        photo: np.ndarray
            A picture of a face of a human.

        correct_emotion: str
            The emotion this human is displaying.

        detector: FER
            The model that will try and recognise an emotion in a picture.

    Return
    ------
        Return True if the function correctly guesses the emotion and a False if not.
    """
    # Change numpy data type in order for the emotion recogniser to accept the image.
    photo = np.array(photo).astype(np.uint8)

    detector_emotion = detector.detect_emotions(photo)

    if not detector_emotion:  # If emotion detector doesn't recognise a face, it returns False.
        return False

    # If detected emotion is correct, return True. If not, return False.
    x = max(detector_emotion[0]['emotions'], key=lambda key: detector_emotion[0]['emotions'][key])
    return correct_emotion == x


def calculate_accuracy(photos: List[np.ndarray], emotions: List[str]) -> float:
    """
    Loop through an amount of photo's. Calculate what percentage of them can correctly be guessed by the detector.
    Parameters
    ----------
        photos: List[np.ndarray]
            A list which contains all the photo's we want to recognise.

        emotions: List[str]
            A list that tells us the correct emotion of a picture.

    Return
    ------
        accuracy: float
            Percentage of photo's that the emotion detector correctly guessed.
    """
    # Activate emotion detector.
    detector = FER(mtcnn=True)

    correct = []

    for i in range(0, len(photos)):  # Does the emotion detector correctly guess the emotion?
        correct.append(recognise_emotion(photos[i], emotions[i], detector))

    # Calculate average correct guesses.
    accuracy = sum(correct) * 100 / len(correct)
    return accuracy
