from models.validation_model import detector_accuracy
from PIL import Image
import numpy as np
from fer import FER
import os

os.chdir(os.getcwd() + '/models/validation_model/data/BB8_validatie/')
print(os.getcwd())


def test_recognise_emotion():
    detector = FER(mtcnn=True)
    filename = 'ThijmeHappy1.png'

    image = np.asarray(Image.open(filename))
    correct_guess = detector_accuracy.recognise_emotion(image, 'happy', detector)

    assert correct_guess


def test_calculate_accuracy():
    photos = [np.asarray(Image.open('ThijmeHappy1.png')), np.asarray(Image.open('MickSad1.png'))]

    accuracy = detector_accuracy.calculate_accuracy(photos, ['happy', 'sad'])

    assert accuracy, 100.0

