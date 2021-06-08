"""Place holder file to save the most used functions."""
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# make a directory if not exists.
if not os.path.exists('Saved-Models'):
    os.makedirs('Saved-Models')


def plot_acc_loss(history: tf.keras.callbacks) -> None:
    """
     Plot the results of a model using the history od it.

     Parameters
     ----------
        history: numpy array
            the callback where all the training results are saved in.
    """
    # Plot accuracy graph
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1.0])
    plt.legend(loc='upper left')
    plt.show()

    # Plot loss graph
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([0, 3.5])
    plt.legend(loc='upper right')
    plt.show()


def save_model_and_weights(model: tf.keras.Sequential, test_acc: float) -> None:
    """
    Save the model and the wights separate to a JSON file.
    this could be used later to test and use the model.

    Parameters
    ----------
        model: tf.keras.Sequential
            the model that should be saved
        test_acc: float
            the results of the test dataset on the model, used to give the model unique name.
    """
    # Serialize and save model to JSON
    test_acc = int(test_acc * 10000)
    model_json = model.to_json()

    with open('Saved-Models\\model' + str(test_acc) + '.json', 'w') as json_file:
        json_file.write(model_json)

    # Serialize and save weights to JSON
    model.save_weights('Saved-Models\\model' + str(test_acc) + '.h5')
    print('Model and weights are saved in separate files.')


def save_all_model(model: tf.keras.Sequential, test_acc: float) -> None:
    """
    Save the whole model using .save from keras.
    this could be used to convert the model to a lit version.

    Parameters
    ----------
        model: tf.keras.Sequential
            the model that should be saved
        test_acc: float
            the results of the test dataset on the model, used to give the model unique name.
    """
    test_acc = int(test_acc * 10000)
    model.save(f'saved_all_model{test_acc}')
    print('Model is saved in a file.')


def save_model_to_lite(test_acc: float) -> None:
    """
    Save the model into a lite version.
    that could be used in the android app.

    Parameters
    ----------
        test_acc: float
            the results of the test dataset on the model, used to give the model unique name.
    """
    test_acc = int(test_acc * 10000)
    path = f'Saved-Models//saved_model{test_acc}'

    # check path to the Saved-Model directory
    if os.path.exists(path):
        converter = tf.lite.TFLiteConverter.from_saved_model(path)
        tflite_model = converter.convert()

        # Save the model
        with open('lite_model.tflite', 'wb') as f:
            f.write(tflite_model)
    else:
        raise OSError('File or directory does not exist!')


def load_model_and_weights(model_path: str, weights_path: str) -> None:
    """
    Load a saved model form the directory and upload the weights to the model.

    Parameters
    ----------
        model_path: str
            string path to the saved model.
        weights_path: str
            string path to the saved wights.
    """
    # check if the file does exists
    if os.path.exists(model_path) and os.path.exists(weights_path):
        # Loading JSON model
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # Loading weights
        model.load_weights(weights_path)
        model.compile(optimizer=tf.keras.optimizers.Adam, loss='binary_crossentropy', metrics=['accuracy'])
        print('Model and weights are loaded and compiled.')
    else:
        raise OSError('File or directory does not exist!')
