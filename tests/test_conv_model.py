"""Tests the model settings while making the model."""
from models.classification_model.conv.conv_model import build_models


def test_build_models():
    """Testing the function by checking if the output is 2 model settings."""
    model_settings = build_models()
    assert len(model_settings) == 2
