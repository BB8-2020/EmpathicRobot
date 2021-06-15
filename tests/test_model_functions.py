"""
Tests for model_functions.py. The fit_model function or tests that use that function are commented out of the code
because it would take too long to run, because the model gets fitted every time.
"""
import pytest

from tensorflow.keras import Sequential
from models.classification_model.conv.conv_model import build_models
from models.classification_model.model_functions import read_data
from models.classification_model.model_functions import fit_model
from models.classification_model.model_functions import compile_model
from models.classification_model.model_functions import evaluate_model


@pytest.fixture
def model_data():
    """Split the data to use in other test functions."""
    x_train, y_train, x_val, y_val, x_test, y_test = read_data("tests/dataclassificationmodel/ferPlus_processed.pbz2", False)
    return x_train, y_train, x_val, y_val, x_test, y_test


@pytest.fixture
def sequential_model():
    """Build a sequential model to test the other functions."""
    model = build_models()
    seq_model = Sequential(model[0]['layers'], name=model[0]['name'])
    return seq_model


@pytest.fixture
def fitted_model(model_data, sequential_model):
    """Fits the model to test the fit function."""
    x_train, y_train, x_val, y_val, x_test, _ = model_data
    compile_model(sequential_model)
    fitted_model = fit_model(sequential_model, 64, 1, False, x_train, y_train, x_val, y_val, x_test)
    return fitted_model


def test_read_data_processed(model_data):
    """Testing the function by checking if the output is the correct size and type."""
    assert len(model_data) == 6 and type(model_data) is tuple


def test_read_data_augmented():
    """Testing the function by checking if the output is the correct size and type."""
    data = read_data("tests/dataclassificationmodel/ferPlus_augment.pbz2", True)
    assert len(data) == 7 and type(data) is tuple


@pytest.mark.long
def test_fit_model(fitted_model):
    """"Testing the function by checking if the loss isn't 0."""
    assert fitted_model.history['loss'] != 0


@pytest.mark.long
def test_compile_model(sequential_model, model_data, fitted_model):
    """Testing the compile_model function by checking if the fit_model function can be run without errors."""
    x_train, y_train, x_val, y_val, x_test, _ = model_data
    try:
        fitted_model
    except RuntimeError:
        pytest.fail("Unexpected RuntimeError: Model needs to be compiled before fitting.")


def test_evaluate_model(sequential_model, model_data):
    """Testing the evaluate_model function by checking the length of the output."""
    _, _, _, _, x_test, y_test = model_data
    compile_model(sequential_model)
    output = evaluate_model(sequential_model, x_test, y_test, 64)
    assert len(output) == 2
