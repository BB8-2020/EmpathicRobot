"""VGG16 layout using keras application."""
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Input
import classification_model_vgg16 as cm_16

""""The implementation below is a simple example of using the file application in keras.
After this, it must be investigated which version and how we want to implement VGG16 exactly."""

# Read the dataset
test_frame = cm_16.read_data("ferPlus_data_json/test.json")

# Create the dataframe
x_train, y_train = cm_16.create_datasets(test_frame, 'formatted_pixels', 'emotion')

# Create model
model = VGG16(input_tensor=Input(shape=(48, 48, 1)), weights=None, classes=7)
print(model.summary())

# This step might be unnecessary if you used a pre-trained model by setting weights = "images'
# Compile model
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# Fit model
model.fit(x_train, y_train, batch_size=120, epochs=1, validation_split=0.2)

# Make a predict!
features = model.predict(x_train[0])
print(features)

"""See https://keras.io/api/applications/vgg/ for more information."""
