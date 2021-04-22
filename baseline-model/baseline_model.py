"""Baseline model using tensorflow. Consists of 3 layers and should only recognize if someone is happy or not."""
from tensorflow import keras
from tensorflow.keras import layers, utils
import pandas as pd
import numpy as np

happy_frame = pd.read_csv("happy_frame.csv")

happy_frame['formatted_pixels'] = ''


def create_trainsets(frame, feature, target):
    """Create and reshape the train sets for the model."""
    x = []
    for index, image_pixels in enumerate(frame[feature]):
        image_string = image_pixels.split(' ')
        x.append(image_string)

    x_train = np.array(x).astype("float32")
    # an image is 48x48 pixels
    X_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    X_train /= 255
    y = frame[target]
    # 2 categories: happy and not happy
    y_train = utils.to_categorical(y, 2)
    return X_train, y_train


def create_model():
    """Create an sequential model that consists of 3 layers."""
    model = keras.Sequential([
        keras.Input(shape=(48, 48, 1)),
        layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu', name='conv1'),
        layers.BatchNormalization(axis=-1),
        layers.Conv2D(kernel_size=(3, 3), filters=128, activation='relu', name='conv2'),
        layers.BatchNormalization(axis=-1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(2)
    ]
    )

    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"],
                  )

    return model


def train_model(model, batch_size, epochs, vs):
    """Train the model using the trainsets created in create_trainsets()."""
    x_train, y_train = create_trainsets(happy_frame, 'pixels', 'happy')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=vs)


if __name__ == "__main__":
    model = create_model()
    print(model.summary())
    train_model(model, 10, 5, 0.2)
