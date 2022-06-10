import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from vit_keras import vit
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
def train():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        'CIFAR-10/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    val_generator = test_datagen.flow_from_directory(
        'CIFAR-10/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
    image_size = 224
    sgd = tf.keras.optimizers.SGD(learning_rate=5e-4)

    callback=keras.callbacks.ModelCheckpoint("vit_cifar10.h5", monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    model = vit.vit_l32(
        image_size=image_size,
        activation='sigmoid',
        pretrained=True,
        include_top=True,
        pretrained_top=False,
        classes=10
    )
    sgd = tf.keras.optimizers.SGD(learning_rate=5e-4)
    model.compile(
        optimizer=sgd,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            'accuracy'
        ]
    )
    model.load_weights("vit_cifar10.h5", by_name=True, skip_mismatch=True)
    model.fit(train_generator,validation_data=val_generator,epochs=10,callbacks=[callback],batch_size=64)
    model.evaluate(val_generator)
if __name__=='__main__':
    train()