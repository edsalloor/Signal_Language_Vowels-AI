import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
print(os.listdir("../"))

train_dir = '../dataset/train'
test_dir = '../dataset/test'


def load_unique():
    size_img = 64,64
    images_for_plot = []
    labels_for_plot = []
    for folder in os.listdir(train_dir):
        for file in os.listdir(train_dir + '/' + folder):
            filepath = train_dir + '/' + folder + '/' + file
            image = cv2.imread(filepath)
            final_img = cv2.resize(image, size_img)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            images_for_plot.append(final_img)
            labels_for_plot.append(folder)
            break
    return images_for_plot, labels_for_plot

images_for_plot, labels_for_plot = load_unique()
print("unique_labels = ", labels_for_plot)


labels_dict = {'A':0,'E':1,'I':2,'O':3,'U':4}


def load_data():
    images = []
    labels = []
    size = 64, 64
    print("LOADING DATA FROM : ", end="")
    for folder in os.listdir(train_dir):
        print(folder, end=' | ')
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            if folder == 'A':
                labels.append(labels_dict['A'])
            elif folder == 'E':
                labels.append(labels_dict['E'])
            elif folder == 'I':
                labels.append(labels_dict['I'])
            elif folder == 'O':
                labels.append(labels_dict['O'])
            elif folder == 'U':
                labels.append(labels_dict['U'])

    images = np.array(images)
    images = images.astype('float32') / 255.0

    labels = keras.utils.to_categorical(labels)  # one-hot encoding

    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.1)

    print()
    print('Loaded', len(X_train), 'images for training,', 'Train data shape =', X_train.shape)
    print('Loaded', len(X_test), 'images for testing', 'Test data shape =', X_test.shape)

    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = load_data()


def build_model():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=(64, 64, 3)))
    model.add(Conv2D(32, kernel_size=3, padding='same', strides=2, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=3, padding='same', strides=2, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=3, padding='same', strides=2, activation='relu'))
    model.add(MaxPool2D(3))

    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
    filepath = "./cnn_model_keras.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    print("MODEL CREATED")
    model.summary()

    callbacks_list = [checkpoint1]
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png', show_shapes=True)

    return model, callbacks_list

model,  callbacks_list = build_model()


def fit_model():
    history = model.fit(X_train, Y_train, batch_size = 64, epochs = 5, validation_split = 0.1,callbacks=callbacks_list)
    return history





model_history = fit_model()

if model_history:
    print('Final Accuracy: {:.2f}%'.format(model_history.history['acc'][4] * 100))
    print('Validation Set Accuracy: {:.2f}%'.format(model_history.history['val_acc'][4] * 100))









