"""Baseline model using tensorflow. Consists of 3 layers and should only recognize if someone is happy or not."""
from tensorflow import keras
from tensorflow.keras import layers, utils
import pandas as pd
import numpy as np

happy_frame = pd.read_csv("happy_frame.csv")

happy_frame['formatted_pixels'] = ''

def format_pixels(happy_frame):
    x = []
    for index, image_pixels in enumerate(happy_frame['pixels']):
        image_string = image_pixels.split(' ')
        x.append(image_string)

    x_train = np.array(x).astype("float32")
    X_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    X_train /= 255
    y = happy_frame['happy']
    y_train = utils.to_categorical(y, 2)
    return X_train, y_train

def create_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(kernel_size=(3, 3), filters=32, input_shape=(48, 48, 1)))
    model.add(layers.BatchNormalization(axis=-1))
    conv1 = layers.Activation("relu")
    model.add(conv1)
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3)))
    model.add(layers.BatchNormalization(axis=-1))
    conv2 = layers.Activation("relu")
    model.add(conv2)
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2))

    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"],
                  )

    return model


def train_model(model):
    X_train, y_train = format_pixels(happy_frame)
    model.fit(X_train, y_train, batch_size=10, epochs=5, validation_split=0.2)


if __name__ == "__main__":
    model = create_model()
    print(model.summary())
    train_model(model)
