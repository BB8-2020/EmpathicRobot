"""Tests for the face detection model."""
import numpy as np
import pytest

from facedetection.face_detection import crop_to_face, reshape_image, convert_to_bytes, face_from_image


def test_crop_to_face():
    """Test the crop_to_face function by checking if the output is the right shape."""
    image = np.load("facedetection/datafacedetection/stormface.npy")
    width = 746
    height = 746
    face = np.array([432, 256, width, height])
    output = crop_to_face(image, face)
    assert np.shape(output) == (width, height, 3)


def test_reshape_image():
    """Test the reshape_image function by checking if the output is of shape 48 x 48."""
    image = np.load("facedetection/datafacedetection/stormface.npy")
    output = reshape_image(image)
    assert np.shape(output) == (48, 48, 3)


def test_convert_to_bytes():
    """Test the convert_to_bytes function by checking if the output type is bytes."""
    image = np.load("facedetection/datafacedetection/stormface.npy")
    output = convert_to_bytes(image)
    assert type(output) is bytes


def test_face_from_image():
    """Test face_from_image function by checking if the image runs all the way to the
       end without errors and checking if the output is type bytes.
    """
    output = face_from_image("facedetection/datafacedetection/stormface.BMP")
    assert type(output) is bytes


def test_raises_exception_on_non_detected_face():
    """Test face_from_image function by checking if the error is raised right. This is when no face is detected."""
    with pytest.raises(Exception) as excinfo:
        face_from_image("facedetection/datafacedetection/false_image.BMP")
    assert "No face was detected in the image" in str(excinfo.value)
