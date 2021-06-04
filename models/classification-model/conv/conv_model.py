from typing import Tuple
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
