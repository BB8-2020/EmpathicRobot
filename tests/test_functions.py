"""Testing the general models functions."""
import pytest
import os

from tensorflow.keras.models import load_model
from models.functions import save_all_model, save_model_to_lite
from models.classification_model.model_functions import read_data, evaluate_model


@pytest.fixture
def prep_for_save():
    """Prepare for the save functions because both need this to be ran."""
    x_train, y_train, x_val, y_val, x_test, y_test = read_data('tests/dataclassificationmodel/ferPlus_processed.pbz2')
    model = load_model("tests/datamodels/model")
    _, test_accuracy = evaluate_model(model, x_test, y_test, 64)
    return model, test_accuracy


@pytest.mark.skip(reason="no way of currently testing this")
def test_save_model_to_lite(prep_for_save):
    """Testing the save_model_to_lite function by checking if the file where the model is saved exists."""
    model, test_accuracy = prep_for_save
    save_model_to_lite(model, test_accuracy)
    assert os.path.exists(f"lite_model{int(test_accuracy * 10000)}.tflite")


@pytest.mark.skip(reason="no way of currently testing this")
def test_save_all_model(prep_for_save):
    """Testing the save_all_model function by checking if the file where the model is saved exists."""
    model, test_accuracy = prep_for_save
    save_all_model(model, test_accuracy)
    assert os.path.exists(f"saved_all_model{int(test_accuracy * 10000)}")
