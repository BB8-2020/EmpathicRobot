from models.validation_model import validation_set as vs
from PIL import Image
import numpy as np
import io
from facedetection.face_detection import face_from_image
import pytest
import os

os.chdir(os.getcwd())
os.chdir('..')
os.chdir('..')

print(os)
os.chdir(os.getcwd() + '/models/validation_model/data/BB8_validatie/')
print(os)


# def test_what_emotion():
#     image = Image.open('ThijmeHappy1.png')
#
#     real_emotion = 'happy'
#     guessed_emotion = vs.what_emotion(image)
#
#     assert real_emotion == guessed_emotion


def test_photo_find_faces_photo():
    path = '/models/validation_model/data/BB8_validatie/'
    filename = 'ThijmeHappy1.png'

    image = face_from_image(path, filename)
    image = Image.open(io.BytesIO(image))
    image = np.array(image)

    new_row = vs.photo_find_faces(path, filename)
    excepted_row = {'Photo': image, 'Correct_emotion': 'emotion'}

    comparison = new_row['Photo'] == excepted_row['Photo']

    assert comparison.all()


def test_photo_find_faces_emotion():
    path = '/models/validation_model/data/BB8_validatie/'
    filename = 'ThijmeHappy1.png'

    new_row = vs.photo_find_faces(path, filename)
    excepted_row = {'Photo': 'photo', 'Correct_emotion': 'happy'}

    assert new_row['Correct_emotion'] == excepted_row['Correct_emotion']

