"""Find faces in a picture, resize that picture and ask the user which emotion they see on this persons face."""
import io
import os

import numpy as np
from PIL import Image

# os.chdir('../..')
from facedetection.face_detection import face_from_image

# os.chdir('models/validation_model/')


def what_emotion(image: Image) -> str:
    """
    User will be shown an image and the program will ask user which emotion is depicted on the image.

    Parameters
    ----------
        image: Image
            This is an PILLOW Image type. It's a picture of a person.

    Return
    ------
        emotion: str
            Return a string of an emotion the user thinks is shown on the picture.
    """
    while True:
        image.show()  # Show user an image.
        print('Neutral, Happy, Anger, Disgust, Surprise, Sad, Fear')
        # What emotion does the person in the picture show?
        emotion = input('Which single of the emotions above you do you recognise?: ').lower()
        print('\n')
        if emotion in ['neutral', 'happy', 'anger', 'disgust', 'surprise', 'sad', 'fear']:
            # Only if the user correctly typed in his emotion, will we continue.
            return emotion


def photo_find_faces(filename: str) -> dict:
    """
    Call face detection function. Get the emotions depicted by the person in the given image.

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

    except Exception:
        # If the face_detector doesn't find a face, we will return a row with None value's. Which we can later remove.
        return {'Photo': None, 'Correct_emotion': None}

    # The image gets returned as a bitmap. We will turn it into a normal picture.
    picture = Image.open(io.BytesIO(picture))

    # Find and open original size picture to show user.
    image = Image.open(os.getcwd() + "/" + filename)

    # Ask user which emotion is shown by the person in the image.
    emotion = what_emotion(image)

    picture = np.asarray(picture)

    # Put photo and emotion in a dictionary and return it.
    new_row = {'Photo': picture, 'Correct_emotion': emotion}
    return new_row
