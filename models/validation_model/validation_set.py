"""Find faces in a picture, resize that picture and ask the user which emotion they see on this persons face."""
import numpy as np
from PIL import Image
import io
import os
os.chdir('../..')
from facedetection.face_detection import face_from_image
os.chdir('models/validation_model/')


def what_emotion(image: Image) -> str:
    """
    The user will be shown an image and the program will ask you which emotion is depicted here.

    Parameters
    ----------
        image: Image
            This is an PILLOW Image type. It's a picture of a person.

    Return
    ------
        emotion: str
            Return a string of an emotion you think is shown on the picture.
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
            The name of the image file.

    Raises
    ------
        Raises error if no face can be detected.

    Return
    ------
        new_row:dict
            You will return a dictionary of the picture and the emotion shows in this image.
    """
    # Get the picture from the path and filename.
    pic_path = os.getcwd() + '/BB8_validation/' + filename

    try:
        picture = face_from_image(pic_path)

    except Exception:
        return {'Photo': None, 'Correct_emotion': None}

    # If the face_detector doesn't find a face, we will skip this picture.
    picture = Image.open(io.BytesIO(picture))

    # Open picture.
    image = Image.open(os.getcwd() + '/BB8_validation/' + filename)

    # Ask user which emotion is shown by the person in the image.
    emotion = what_emotion(image)

    picture = np.asarray(picture)

    # Put the photo and the emotion in a dictionary.
    new_row = {'Photo': picture, 'Correct_emotion': emotion}
    return new_row
