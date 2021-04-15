"""Baseline model using tensorflow. Consists of 3 layers and should only recognize if someone is happy or not."""

from tensorflow import keras
from tensorflow.keras import layers

# Layer 1 receives the input which is an image of 48x48 pixels
baseline_model_input = keras.Input(shape=(48, 48, 3))
# Layer 2 has a filter of 24x24x3
x = layers.Conv2D(24, 3, activation="relu")(baseline_model_input)
# Layer 3 separates the output into 2 possible classes
baseline_model_output = layers.Dense(2)(x)

model = keras.Model(inputs=baseline_model_input, outputs=baseline_model_output, name="Baseline model")

model.compile(loss=keras.losses.SparseCategoricalCrossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=["accuracy"],
              )

print(model.summary())
