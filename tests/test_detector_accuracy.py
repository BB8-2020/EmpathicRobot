"""Tests for the detector_accuracy.py functions."""
import pytest
from fer import FER
from PIL import Image
from models.validation_model.detector_accuracy import recognise_emotion, calculate_accuracy


def test_recognise_emotion():
    """Test recognise_emotion function by checking if the output is True."""
    detector = FER(mtcnn=True)
    image = Image.open("tests/datavalidation/storm_surprise.jpg")
    output = recognise_emotion(image, "surprise", detector)
    assert output is True


def test_calculate_accuracy():
    """Test calculate_accuracy function by checking the right output."""
    image = Image.open("tests/datavalidation/storm_surprise.jpg")
    output = calculate_accuracy([image], ["surprise"])
    assert output == 100
