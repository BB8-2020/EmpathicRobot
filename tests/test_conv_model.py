"""Tests the model settings while making the model."""
from models.classification_model.conv.conv_model import build_models


def test_build_models():
    model_settings = build_models()
    assert len(model_settings) == 2
