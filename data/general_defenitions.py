import bz2
import numpy as np
from math import ceil
import _pickle as cPickle
import matplotlib.pyplot as plt
from sklearn import model_selection
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def comp_pickle_save(data, filename):
    with bz2.BZ2File(filename, 'w') as f:
        cPickle.dump(data, f)
      
    
def split_data(X, y):
    test_size = ceil(len(X) * 0.1)

    # Split Data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=test_size,
                                                                      random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test


def data_augmentation(x_train):
    shift = 0.1
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        height_shift_range=shift,
        width_shift_range=shift)
    datagen.fit(x_train)
    return datagen
    
    
def show_images(x_train, y_train, datagen=None):
    if datagen != None:
        it = datagen.flow(x_train, y_train, batch_size=1)
    plt.figure(figsize=(10, 7))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if datagen != None:
            plt.imshow(np.squeeze(it.next()[0][0]), cmap='gray')
        else:
            plt.imshow(np.squeeze(x_train[i]), cmap='gray')
            
        if type(y_train[i]) == str:
            plt.xlabel(y_train[i])
    plt.show()