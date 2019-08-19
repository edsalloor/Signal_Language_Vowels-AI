
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file( 'cnn_model_keras.h5')
tfmodel = converter.convert()
open ("model.tflite" , "wb") .write(tfmodel)