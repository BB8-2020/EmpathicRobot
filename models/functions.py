"""Place holder file to save the most used functions."""
import os

import matplotlib.pyplot as plt
import tensorflow as tf

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


def save_model_to_lite(model: tf.keras.Sequential, test_acc: float) -> None:
    """
    Save the model into a lite version.
    that could be used in the android app.

    Parameters
    ----------
        model: tf.keras.Sequential
            the model that sould be saved as a lite model.
        test_acc: float
            the results of the test dataset on the model, used to give the model unique name.

    """
    test_acc = int(test_acc * 10000)

    # check path to the Saved-Model directory

    converter = tf.lite.TFLiteConverter.from_keras.model(model)
    tflite_model = converter.convert()

    # Save the model
    with open(f'lite_model{test_acc}.tflite', 'wb') as f:
        f.write(tflite_model)
