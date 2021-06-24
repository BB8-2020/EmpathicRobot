"""Tests for the baseline_model functions."""
import pytest

from src.models.baseline_model.baseline_model import read_data
from src.models.baseline_model.baseline_model import create_datasets
from src.models.baseline_model.baseline_model import create_model
from src.models.baseline_model.baseline_model import train_model
from src.models.baseline_model.baseline_model import compile_model
from src.models.baseline_model.baseline_model import evaluate_model


@pytest.fixture
def run_read_data_train():
    """Read the given datafile."""
    data = read_data("src/tests/databaselinemodel/train_happy_frame.json")
    return data


@pytest.fixture
def run_create_datasets(run_read_data_train):
    """Read data and creates the datasets with that data."""
    frame = run_read_data_train
    x_feature, y_target = create_datasets(frame, 'formatted_pixels', 'happy')
    return x_feature, y_target


@pytest.fixture
def run_train_model(run_read_data_train):
    """Read data, creates a model, compiles that model and trains that model."""
    data = run_read_data_train
    model = create_model()
    compile_model(model)
    history = train_model(model, data, batch_size=64, epochs=1, vs=0.2, save=False)
    return model, history


def test_read_data(run_read_data_train):
    """Testing the read_data function by checking if the column names are correct."""
    data = run_read_data_train
    assert "pixels" in data.keys() and "happy" in data.keys() and "formatted_pixels" in data.keys()


def test_raises_oserror_read_data():
    """Testing the raise Exception in the read_data function by giving a wrong path."""
    with pytest.raises(Exception, match=r"File in this .* does not exist"):
        read_data("src/tests/databaselinemodel")


# All tests marked with long take to long to run. If you want to run these tests use . pytest -vs
@pytest.mark.long
def test_create_datasets(run_create_datasets):
    """Testing create_datasets() by checking if the reshape and categorization was successful."""
    x_feature, y_target = run_create_datasets

    # width and height of images
    width = x_feature.shape[1]
    heigth = x_feature.shape[2]
    assert [width, heigth] == [48, 48] and y_target.shape[1] == 2


def test_create_model():
    """Testing create_model() by checking if the amount of layers is correct."""
    model = create_model()
    assert len(model.layers) == 7


@pytest.mark.long
def test_train_model(run_train_model):
    """Testing if the model got trained by checking if the loss isn't 0."""
    _, history = run_train_model
    assert history.history['loss'] != 0


@pytest.mark.long
def test_compile_model(run_train_model):
    """
        Testing if compile_model() compiles the model.
        This is done by calling the train_model function and see if the function can run without errors.
    """
    try:
        run_train_model
    except RuntimeError:
        pytest.fail("Unexpected RuntimeError: Model needs to be compiled before training.")


@pytest.mark.long
def test_evaluate_model(run_train_model):
    """Testing the evaluate_model function by checking the output length and type."""
    test_data = read_data("src/tests/databaselinemodel/test_happy_frame.json")
    model, _ = run_train_model
    output = evaluate_model(model=model, frame=test_data, batch_size=256)
    assert len(output) == 2 and type(output[0]) == float and type(output[1]) == float
