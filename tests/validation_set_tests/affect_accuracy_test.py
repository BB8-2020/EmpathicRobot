from models.validation_model import affect_accuracy
from PIL import Image
import numpy as np
import pandas as pd
import os

os.chdir('../../models/validation_model/data/BB8_validatie/')


def test_update_df():
    df1 = pd.DataFrame(columns=['image', 'emotion'])

    data = [[[1, 2, 3, 4]], ['disgust']]

    df1 = affect_accuracy.update_df(0, 1, data, df1)

    df2 = pd.DataFrame({'image': [[1, 2, 3, 4]], 'emotion': ['disgust']}, columns=['image', 'emotion'])
    assert df1.equals(df2)


def test_aff_accuracy():
    df1 = pd.DataFrame(columns=['image', 'emotion'])

    image = np.asarray(Image.open('ThijmeHappy1.png'))

    data = [[[image]], ['happy']]

    accuracy = affect_accuracy.aff_accuracy(0, 1, data, df1)

    assert accuracy, 100.0



