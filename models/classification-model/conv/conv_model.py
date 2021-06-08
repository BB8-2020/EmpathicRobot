import _pickle as cPickle
import bz2
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization


def build_models(input_shape: Tuple[int, int, int] = (48, 48, 1), num_classes: int = 7):
    num_features = 64
    activation_ = 'relu'

    models_settings = [{
        "name": "Version_1",
        "layers": [
                # 1st stage
                Conv2D(num_features, kernel_size=(3, 3), input_shape=input_shape, activation=activation_),
                BatchNormalization(), Dropout(0.5),
                # 2nd stage
                Conv2D(num_features, (3, 3), activation=activation_),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                # 3rd stage
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                # 4th stage
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                # 5th stage
                Conv2D(4 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                # Fully connected neural networks
                Flatten(), Dense(1024, activation=activation_), Dropout(0.2),
                Dense(1024, activation=activation_), Dropout(0.2), Dense(num_classes, activation='softmax')
                ]},

        {
        "name": "Version_2",
        "layers": [
                # 1st stage
                Conv2D(num_features, kernel_size=(3, 3), input_shape=input_shape, activation=activation_),
                BatchNormalization(),
                Conv2D(num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                Dropout(0.5),
                # 2nd stage
                Conv2D(num_features, (3, 3), activation=activation_),
                Conv2D(num_features, (3, 3), activation=activation_),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                # 3rd stage
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                # 4rd stage
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                Conv2D(2 * num_features, kernel_size=(3, 3), activation=activation_),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(4 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                # 5th stage
                Conv2D(4 * num_features, kernel_size=(3, 3), activation=activation_),
                BatchNormalization(),
                # Fully connected neural networks
                Flatten(), Dense(1024, activation=activation_), Dropout(0.2),
                Dense(1024, activation=activation_), Dropout(0.2), Dense(num_classes, activation='softmax')
        ]
    }]
    return models_settings


def read_data(path: str, datagen: bool = False):
    if datagen:
        datagen, x_train, y_train, x_val, y_val, x_test, y_test = cPickle.load(bz2.BZ2File(str(path), 'rb'))
        return datagen, x_train, y_train, x_val, y_val, x_test, y_test

    else:
        x_train, y_train, x_val, y_val, x_test, y_test = cPickle.load(bz2.BZ2File(str(path), 'rb'))
        return x_train,  y_train, x_val, y_val, x_test, y_test


def fit_model(model: keras.Sequential, batch_size: int = 64, epochs: int = 100, dategen: bool = False, *data):

    if dategen:
        datagen, x_train, y_train, x_val, y_val, x_test= data
        history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            steps_per_epoch=len(x_train) // batch_size,
                            validation_data=(x_val, y_val), verbose=2)
    else:
        x_train, y_train, x_val, y_val, x_test = data

        history = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs,
                            steps_per_epoch=len(x_train) // batch_size,
                            validation_data=(x_val, y_val), verbose=2)
    return history


def compile_model(model: keras.Sequential):

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


def evaluate_model(model: keras.Sequential, x_test, y_test, batch_size: int = 64):
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    return test_loss, test_acc


def main():
    pass