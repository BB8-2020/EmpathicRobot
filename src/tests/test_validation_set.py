"""Tests for the validation_set.py functions."""
import pytest
import mock
import builtins
from PIL import Image

from src.models.validation_model.validation_set import choose_emotion, label_faces


def test_choose_emotion():
    """Test the what_emotion function by checking the return value."""
    image = Image.open("src/tests/datavalidation/storm_surprise.jpg")
    with mock.patch.object(builtins, 'input', lambda _: 'surprise'):
        assert choose_emotion(image) == 'surprise'


@pytest.mark.skip(reason="There is no data file added to the repository.")
def test_label_faces():
    """Test the photo_find_faces function by checking the output type and output keys."""
    output = label_faces("storm_surprise.jpg")
    assert type(output) == dict and list(output.keys()) == ["Photo", "Correct_emotion"]


def test_photo_find_faces_exception():
    """Test the exception in photo_find_faces() by checking the values of the output."""
    output = label_faces("src/tests/datafacedetection/false_image.BMP")
    assert list(output.values()) == [None, None]
