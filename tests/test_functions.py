"""Testing the general models functions."""
import pytest
import os
from tensorflow.keras import Sequential
from matplotlib import pyplot as plt

from models.functions import plot_acc_loss, save_all_model, save_model_to_lite
from models.classification_model.model_functions import read_data, fit_model, compile_model, evaluate_model
from models.classification_model.conv.conv_model import build_models


@pytest.fixture
def everything_to_fit():
    """All test functions need these steps so its done in this function."""
    x_train, y_train, x_val, y_val, x_test, y_test = read_data('tests/dataclassificationmodel/ferPlus_processed.pbz2')
    models = build_models()
    model = Sequential(models[0]['layers'], name=models[0]['name'])
    compile_model(model)
    history = fit_model(model, 64, 1, False, x_train, y_train, x_val, y_val, x_test)
    return history, model, x_test, y_test


@pytest.fixture
def prep_for_save(everything_to_fit):
    """Preparation for the save functions because both need this to be ran."""
    _, model, x_test, y_test = everything_to_fit
    _, test_accuracy = evaluate_model(model, x_test, y_test, 64)
    return model, test_accuracy


@pytest.mark.long
def test_plot_acc_loss(everything_to_fit):
    """Testing the plot_acc_loss function by checking if the current figure number is 1."""
    history, _, _, _ = everything_to_fit
    plt.close()
    plot_acc_loss(history)
    plt.close("all")
    assert plt.gcf().number == 1


@pytest.mark.long
def test_save_model_to_lite(prep_for_save):
    """Testing the save_model_to_lite function by checking if the file where the model is saved exists."""
    model, test_accuracy = prep_for_save
    save_model_to_lite(model, test_accuracy)
    assert os.path.exists(f"lite_model{int(test_accuracy * 10000)}.tflite")


@pytest.mark.long
def test_save_all_model(prep_for_save):
    """Testing the save_all_model function by checking if the file where the model is saved exists."""
    model, test_accuracy = prep_for_save
    save_all_model(model, test_accuracy)
    assert os.path.exists(f"saved_all_model{int(test_accuracy * 10000)}")
