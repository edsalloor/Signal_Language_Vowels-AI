import numpy as np
import pickle
import cv2
import os
import keras
import matplotlib.pyplot as plt
from glob import glob
from keras import optimizers
from keras.layers import Input, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint
from keras import backend as k

from keras.preprocessing.image import ImageDataGenerator


train_path = 'D:/Descargas/NBS2/train'
valid_path = 'D:/Descargas/NBS2/valid'
test_path = 'D:/Descargas/NBS2/test'

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
k.set_image_dim_ordering('tf')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
	img = cv2.imread(train_path+'/A/a_men (1).JPG', 0)
	return img.shape

def get_num_of_classes():
	return len(glob(train_path+'/'))


def getClasses(thedir):
	return [ name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name)) ]

classes= getClasses(train_path)
print(classes)
print(get_num_of_classes())
image_x, image_y = get_image_size()

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')




def vgg16_model():

	#vgg16_model = keras.applications.vgg16.VGG16(include_top=False, classes=classes, input_shape=(image_x, image_y, 3))
	vgg16_model = keras.applications.vgg16.VGG16(include_top=False, classes=5, input_shape=(image_x, image_y, 3))

	model = Sequential()
	for layer in vgg16_model.layers:
		model.add(layer)

	model.summary()
	model.layers.pop()
	model.summary()
	for layer in model.layers:
		layer.trainable = False
	model.add(Flatten())
	model.add(Dense(5, activation='softmax'))
	model.summary()
	model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
	filepath = "../model/cnn_model_keras.h5"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]

	return model, callbacks_list





def cnn_model2():

	model = Sequential()
	model.add(Conv2D(64, (3, 3),input_shape=(image_x, image_y, 3),
					  activation='relu',
					  padding='same'))
	model.add(Conv2D(64, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block1_conv2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

	# Block 2
	model.add(Conv2D(128, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block2_conv1'))
	model.add(Conv2D(128, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block2_conv2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

	# Block 3
	model.add(Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv1'))
	model.add(Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv2'))
	model.add(Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

	# Block 4
	model.add(Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block4_conv1'))
	model.add(Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block4_conv2'))
	model.add(Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block4_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

	# Block 5
	model.add(Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block5_conv1'))
	model.add(Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block5_conv2'))
	model.add(Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block5_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
	for layer in model.layers:
		layer.trainable = False
	model.add(Flatten())
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(5, activation='softmax'))
	model.summary()
	model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
	filepath = "../model/cnn_model_keras.h5"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]

	return model, callbacks_list

def cnn_model():

	model = Sequential()
	model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
	model.add(Conv2D(64, (5,5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(5, activation='softmax'))
	Adam = optimizers.Adam(lr=.0001)
	model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
	filepath="../model/cnn_model_keras.h5"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]
	#from keras.utils import plot_model
	#plot_model(model, to_file='model.png', show_shapes=True)



	return model, callbacks_list

def train():
	# create a data generator

	# load and iterate training dataset
	train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(image_x, image_y), classes=classes, batch_size=20)
	# load and iterate validation dataset
	valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(image_x, image_y), classes=classes, batch_size=7)
	# load and iterate test dataset
	test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(image_x, image_y), classes=classes, batch_size=5)


	model, callbacks_list = cnn_model()
	model.summary()
	model.fit_generator(train_batches, validation_data=valid_batches, epochs=30, steps_per_epoch=6, validation_steps=4,verbose=1, callbacks=callbacks_list)
	predictions=model.predict_generator(valid_batches,steps=1,verbose=0)

	print(predictions)
	#scores = model.evaluate(valid_batches, verbose=0)
	#print("CNN Error: %.2f%%" % (100-scores[1]*100))
	#model.save('cnn_model_keras2.h5')



train()


k.clear_session();