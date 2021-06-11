import pytest

from tensorflow.keras import Sequential
from models.classification_model.conv.conv_model import build_models
from models.classification_model.model_functions import read_data
from models.classification_model.model_functions import fit_model
from models.classification_model.model_functions import compile_model
from models.classification_model.model_functions import evaluate_model

def test_read_data():
    """Testing the function by checking if the output is the correct size and type."""
    output = read_data("tests/dataclassificationmodel/ferPlus_processed", False)
    assert len(output) == 6 and type(output) is tuple

def test_fit_model():
    """"Testing the function by checking if the loss isn't 0."""
    model = build_models()
    data = read_data("tests/dataclassificationmodel/ferPlus_processed", False)
    seq_model = Sequential(model[0]['layers'], name=model[0]['name'])
    compile_model(seq_model)
    output = fit_model(seq_model, 64, 1, False, data[0], data[1], data[2], data[3], data[4])
    assert output.history['loss'] != 0

def test_compile_model():
    """Testing the compile_model function by checking if the fit_model function can be run without errors."""
    model = build_models()
    data = read_data("tests/dataclassificationmodel/ferPlus_processed", False)
    seq_model = Sequential(model[0]['layers'], name=model[0]['name'])
    compile_model(seq_model)
    try:
        fit_model(seq_model, 64, 1, False, data[0], data[1], data[2], data[3], data[4])
    except RuntimeError:
        pytest.fail("Unexpected RuntimeError. Model needs to be compiled before fitting.")

def test_evaluate_model():
    """Testing the evaluate_model function by checking the length of the output."""
    model = build_models()
    data = read_data("tests/dataclassificationmodel/ferPlus_processed", False)
    seq_model = Sequential(model[0]['layers'], name=model[0]['name'])
    compile_model(seq_model)
    fit_model(seq_model, 64, 1, False, data[0], data[1], data[2], data[3], data[4])
    output = evaluate_model(seq_model, data[4], data[5], 64)
    assert len(output) == 2
