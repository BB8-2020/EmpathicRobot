"""Tests the model settings while making the model."""
import pytest
from models.classification_model.conv.conv_model import build_models


def test_build_models():
    model_settings = build_models()
    assert len(model_settings) == 2

def test_read_data():
    pass

def test_fit_model():
    pass

def test_compile_model():
    pass

def test_evaluate_model():
    pass