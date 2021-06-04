import os
import _pickle as cPickle
import bz2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from models.functions import *


def define_model(input_shape=(48, 48, 1), classes=7):
    num_features = 64

    model = Sequential()

    # 1st stage
    model.add(Conv2D(num_features, kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.5))

    # 2nd stage
    model.add(Conv2D(num_features, (3, 3), activation='relu'))
    model.add(Conv2D(num_features, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 3rd stage
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    # 4th stage
    model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 5th stage
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Flatten())

    # Fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(classes, activation='softmax'))
    return model


def build_model_1(input_shape=(48, 48, 1), classes=7):
    num_features = 64

    model = Sequential(name="Model_1.0")

    # 1st stage
    model.add(Conv2D(num_features, kernel_size=(3, 3), input_shape=input_shape, name="1"))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    # model.add(Conv2D(num_features, kernel_size=(3, 3)))
    # model.add(BatchNormalization())
    # model.add(Activation(activation='relu'))
    model.add(Dropout(0.5))

    # 2nd stage
    model.add(Conv2D(num_features, (3, 3), activation='relu', name='2'))
    # model.add(Conv2D(num_features, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 3rd stage
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), name='3'))
    model.add(BatchNormalization())
    # model.add(Activation(activation='relu'))
    # model.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    # model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    # 4th stage
    model.add(Conv2D(2 * num_features, (3, 3), activation='relu', name='4'))
    # model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 5th stage
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), name='5'))
    # model.add(BatchNormalization())
    # model.add(Activation(activation='relu'))
    # model.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    model.add(Flatten())

    # Fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(classes, activation='softmax'))

    return model




def run_model(*data):
    fer_classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

#     X, y = preprocess_data()
#     X, y = clean_data_and_normalize(X, y)
#     x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, y)
#     datagen = data_augmentation(x_train)

    x_train, y_train, x_val, y_val, x_test, y_test = data
    epochs = 1
    batch_size = 64

    print("X_train shape: " + str(x_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(x_test.shape))
    print("Y_test shape: " + str(y_test.shape))
    print("X_val shape: " + str(x_val.shape))
    print("Y_val shape: " + str(y_val.shape))

    # Training model from scratch
    # model = define_model(input_shape=x_train[0].shape, classes=len(fer_classes))
    model = build_model_1(input_shape=x_train[0].shape, classes=len(fer_classes))

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    # history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
    #                     steps_per_epoch=len(x_train) // batch_size,
    #                     validation_data=(x_val, y_val), verbose=2)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        steps_per_epoch=len(x_train) // batch_size,
                        validation_data=(x_val, y_val), verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)


    save_model_summary(model, test_acc)
    # plot_acc_loss(history)
    # save_model_and_weights(model, test_acc)
    # save_all_model(model, test_acc)
    # save_model_to_lite(test_acc)

# os.chdir(os.getcwd() + '/data/')
# run_model(*cPickle.load(bz2.BZ2File('ferPlus_processed', 'rb')))