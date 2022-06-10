import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import cifar10,cifar100
import PIL.Image
import tensorflow_datasets as tfds
from res_mlp import ResMLP12,ResMLP24,ResMLP36
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
num_classes=10
'''
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'data/sorted/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

val_generator = test_datagen.flow_from_directory(
        'data/sorted/valid',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
'''

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
def train():

    layer1 = tf.keras.layers.Resizing( 224, 224, interpolation='bilinear', crop_to_aspect_ratio=True,name='resize')
    inputs = tf.keras.Input(shape=(32,32,3),name='input')
    model = ResMLP12(num_classes=num_classes)
    #model = ResMLP24(num_classes=num_classes)
    #model = ResMLP36(num_classes=num_classes)
    x = layer1(inputs)
    outputs = model(x)
    mlp = tf.keras.Model(inputs, outputs)
    adam = tf.keras.optimizers.Adam(learning_rate=5e-4)
    sgd = tf.keras.optimizers.SGD(learning_rate=5e-4)

    callback=keras.callbacks.ModelCheckpoint("resmlp12_silu.h5", monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    mlp.compile(
        optimizer=sgd,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            'accuracy'
        ]
    )
    mlp.fit(x_train,y_train,validation_data=(x_train,y_train),epochs=10,callbacks=[callback],batch_size=64)
    
if __name__=='__main__':
    train()