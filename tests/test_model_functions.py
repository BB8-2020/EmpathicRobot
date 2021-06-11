"""Tests for model_functions.py."""
import pytest

from tensorflow.keras import Sequential
from models.classification_model.conv.conv_model import build_models
from models.classification_model.model_functions import read_data
from models.classification_model.model_functions import fit_model
from models.classification_model.model_functions import compile_model
from models.classification_model.model_functions import evaluate_model

# every function needs this so we run it ones and use the output in the tests
model = build_models()
data = read_data("tests/dataclassificationmodel/ferPlus_processed.pbz2", False)
seq_model = Sequential(model[0]['layers'], name=model[0]['name'])
compile_model(seq_model)
fitted_model = fit_model(seq_model, 64, 1, False, data[0], data[1], data[2], data[3], data[4])

def test_read_data_processed():
    """Testing the function by checking if the output is the correct size and type."""
    global data
    assert len(data) == 6 and type(data) is tuple


def test_read_data_augmented():
    """Testing the function by checking if the output is the correct size and type."""
    data = read_data("tests/dataclassificationmodel/ferPlus_augment.pbz2")
    assert len(data) == 6 and type(data) is tuple


def test_fit_model():
    """"Testing the function by checking if the loss isn't 0."""
    global fitted_model
    assert fitted_model.history['loss'] != 0


def test_compile_model():
    """Testing the compile_model function by checking if the fit_model function can be run without errors."""
    global seq_model, data
    try:
        fit_model(seq_model, 64, 1, False, data[0], data[1], data[2], data[3], data[4])
    except RuntimeError:
        pytest.fail("Unexpected RuntimeError: Model needs to be compiled before fitting.")


def test_evaluate_model():
    """Testing the evaluate_model function by checking the length of the output."""
    global seq_model, data
    output = evaluate_model(seq_model, data[4], data[5], 64)
    assert len(output) == 2
