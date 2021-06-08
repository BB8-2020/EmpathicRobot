"""VGG16 layout using keras application."""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Input
import tensorflow as tf
import vgg16 as cm_16

""""The implementation below is a simple example of using the file application in keras.
After this, it must be investigated which version and how we want to implement VGG16 exactly."""

# Read the dataset
test_frame = cm_16.read_data("test_expand.pkl")

# Create the dataframe
x_train, y_train = cm_16.create_datasets(test_frame, 'formatted_pixels', 'emotion')

# Convert the grayscale images to RGB images.
# Necessary for the pre-trained model
x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train), name=None)

# Create model
model = VGG16(include_top=False, weights="imagenet", input_tensor=Input(shape=(128, 128, 3)))
print(model.summary())

# This step might be unnecessary if you used a pre-trained model by setting weights = "images'
# Compile model
# model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# Fit model
# model.fit(x_train, y_train, batch_size=120, epochs=1, validation_split=0.2)

# Make a predict!
features = model.predict(x_train[0])
print(features)

"""See https://keras.io/api/applications/vgg/ for more information."""
