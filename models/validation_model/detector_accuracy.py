"""In this document there are functions that will go through a series of photo's and will try to guess the correct
emotion depicted there."""
import numpy as np
from typing import List
from fer import FER


def recognise_emotion(photo: np.ndarray, correct_emotion: str, detector: FER) -> bool:
    """
    Give our detector an image. Return whether or not our detector has guessed the correct emotion depicted by the
    person in the image.

    Parameters
    ----------
        photo: np.ndarray
            A picture of a face of a human.

        correct_emotion: str
            The emotion that this human is displaying.

        detector: FER
            The model that will try and recognise an emotion in a picture.



    Return
    ------
        Return True if the function correctly guesses the emotion and a False if not.
    """

    # The detector says our picture is of the wrong datatype. So we will change it. But dont worry, the picture
    # will stay the same.
    photo = np.array(photo).astype(np.uint8)

    detector_emotion = detector.detect_emotions(photo)

    if not detector_emotion:  # If the detector doesn't recognise a face, it will return a False.
        return False

    # If the detected emotion is the correct one, it will return True. If not, it will return False.
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
            Return a number that tells the user what percentage of the photo's the model correctly recognised.
    """
    # Activate the emotion detector.
    detector = FER(mtcnn=True)

    correct = []

    for i in range(0, len(photos)):  # Does the detector correctly guess the emotion?
        correct.append(recognise_emotion(photos[i], emotions[i], detector))

    # Calculate average correct guesses.
    accuracy = sum(correct) * 100 / len(correct)
    return accuracy
