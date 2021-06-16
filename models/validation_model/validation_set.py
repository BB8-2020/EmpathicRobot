"""
The purpose of this file is to find faces in a picture, resize that picture and ask the user which emotion they see
on this persons face.
"""

import numpy as np
from PIL import Image
import io
from facedetection.face_detection import face_from_image


def what_emotion(image: Image) -> str:
    """
    You will see an image and the program will ask you which emotion you see depicted here.

    Parameters
    ----------
        image: Image
            This is an PILLOW Image type. It's a picture of a person.

    Return
    ------
        emotion: str
            Return a string of an emotion you think is shown on the picture.
    """

    # This while loop will show you the image and will ask for one of the 7 emotions. As long as a real answer hasn't
    # been given, the loop will continue.
    while True:
        image.show()
        print('Neutral, Happy, Anger, Disgust, Surprise, Sad, Fear')
        emotion = input('Which single of the emotions above you do you recognise?: ').lower()
        print('\n')
        if emotion in ['neutral', 'happy', 'anger', 'disgust', 'surprise', 'sad', 'fear']:
            return emotion


def photo_find_faces(path: str, filename: str) -> dict:
    """
    This function takes a picture and tries to find a face. If it does, it will take a picture of only the face and
    crops it into an 48 pixels by 48 pixels array. You will see the whole image and the program will ask you which
    emotion is depicted here.

    Parameters
    ----------
        path: str
            The path to the image file

        filename: str
            The name of the image file

    Return
    ------
        new_row:dict
            You will return a dictionary of the picture and the emotion shows in this image
    """

    # Get the picture from the path and filename.
    picture = face_from_image(path, filename)

    # If the face_detector doesn't find a face, we will skip this picture.
    if picture is None:
        return {'Photo': None, 'Correct_emotion': None}

    picture = Image.open(io.BytesIO(picture))

    image = Image.open(path + filename)

    # What emotion is shown in the image.
    emotion = what_emotion(image)

    picture = np.asarray(picture)

    # Put the photo and the emotion in a dictionary.
    new_row = {'Photo': picture, 'Correct_emotion': emotion}
    return new_row




