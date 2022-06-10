import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import cifar10
from distiller import Distiller
from res_mlp_teacher import ResMLP12_teacher
from res_mlp_student import ResMLP12_student
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)
LR = 0.000001
WEIGHT_DECAY = 0.001
BATCH_SIZE = 32
EPOCH = 300

import PIL.Image
import tensorflow_datasets as tfds
import numpy as np






train_ds = tf.keras.utils.image_dataset_from_directory(
        'imagenet/train',
        seed=123,
        label_mode='categorical',
        image_size=(224,224),
        batch_size=64)
val_ds = tf.keras.utils.image_dataset_from_directory(
        'imagenet/val',
        seed=123,
        label_mode='categorical',
        image_size=(224,224),
        batch_size=64)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)



def train():
    teacher = ResMLP12_teacher(input_shape=(224, 224, 3), num_classes=1000)
    
    
    teacher.compile(metrics=['accuracy'])
    
    
    student = ResMLP12_student(input_shape=(224, 224, 3), num_classes=1000)
    
    
    adam = tf.keras.optimizers.Adam(learning_rate=5e-4)
    sgd = tf.keras.optimizers.SGD(learning_rate=5e-4)
    student.compile(
        optimizer=sgd,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy'
        ]
    )
    
    
    
    distiller = Distiller(student=student,teacher=teacher)
    distiller.compile(
        optimizer=adam,
        student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=3,
        metrics=[
            'accuracy'
        ]
        
    )
    
    
    distiller.fit(train_ds,
    validation_data=val_ds, 
    epochs=10
    )
    
if __name__ == "__main__":
    train()