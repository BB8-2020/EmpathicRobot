"""Find faces in a picture, resize that picture and ask the user which emotion they see on this persons face."""
import io
import os

import numpy as np
from PIL import Image
from src.facedetection.face_detection import face_from_image
from src.facedetection.exceptions import NoFace


def choose_emotion(image: Image) -> str:
    """
    User will be shown an image and the program will ask user which emotion is depicted on the image.

    Parameters
    ----------
        image: Image
            This is an PILLOW Image type.

    Return
    ------
        emotion: str
            Return a string of an emotion the user thinks is shown on the picture.
    """
    emotions = ['neutral', 'happy', 'anger', 'disgust', 'surprise', 'sad', 'fear']
    while True:
        image.show()  # Show user an image.
        print(emotions)
        emotion = str(input('Which single of the emotions above you do you recognise?: \n')).lower()
        if emotion in emotions:
            return emotion


def label_faces(filename: str) -> dict:
    """
    Get face from image. Get the emotions depicted by the person in the given image.
    And label the face with the chosen emotion.

    Parameters
    ----------
        filename: str
            Name of the image file.

    Return
    ------
        new_row:dict
            Return a dictionary of the picture and the emotion shown in this image.
    """
    # Get the picture from the path and filename.
    pic_path = os.getcwd() + "/" + filename

    try:
        picture = face_from_image(pic_path)

    except NoFace:
        # If the face_detector doesn't find a face, we will return a row with None value's. Which we can later remove.
        return {'Photo': None, 'Correct_emotion': None}

    # The image gets returned as a bitmap. We will turn it into a normal picture.
    picture = Image.open(io.BytesIO(picture))

    # Find and open original size picture to show user.
    image = Image.open(os.getcwd() + "/" + filename)

    # Ask user which emotion is shown by the person in the image.
    emotion = choose_emotion(image)

    picture = np.asarray(picture)

    # Put photo and emotion in a dictionary and return it.
    new_row = {'Photo': picture, 'Correct_emotion': emotion}
    return new_row
