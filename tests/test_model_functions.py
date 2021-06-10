import pytest

from tensorflow.keras import Sequential
from models.classification_model.conv.conv_model import build_models
from models.classification_model.model_functions import read_data
from models.classification_model.model_functions import fit_model
from models.classification_model.model_functions import compile_model

def test_read_data():
    output = read_data("dataclassificationmodel/ferPlus_processed", False)
    assert len(output) == 6 and type(output) is tuple

def test_fit_model():
    model = build_models()
    data = read_data("dataclassificationmodel/ferPlus_processed", False)
    model1 = Sequential(model[0]['layers'], name=model[0]['name'])
    compile_model(model1)
    output = fit_model(model1, 64, 1, False, data[0], data[1], data[2], data[3], data[4])
    print(output)
    pass

def test_compile_model():
    pass

def test_evaluate_model():
    pass