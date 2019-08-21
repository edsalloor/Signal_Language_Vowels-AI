import os
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np 
import keras
import cv2
from sklearn.model_selection import train_test_split

print(os.listdir("./"))

# Cargar los pesos (parámetros) que produjeron la mejor precisión en validación
model = load_model('./cnn_model_keras.h5')
train_dir = '../dataset/train'
test_dir = '../dataset/test'

# 11. Calcular la precisión en clasificación en el conjunto de prueba

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
X_predict = X_test[-10:,:]

score = model.evaluate(X_test, Y_test, verbose=0)
accuracy = 100*score[1]

# mostrar la precisión en prubea
print('Precisión durante la prueba: %.4f%%' % accuracy)

result=model.predict(X_predict)
labels = (result > 0.5).astype(np.int)
print(labels)

# graficar las 4 primeras imágenes nuevas
fig = plt.figure(figsize=(5,5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(X_predict[i], cmap='gray')
plt.show()