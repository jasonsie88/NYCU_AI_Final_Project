import tensorflow as tf
import numpy as np
from tensorflow import keras 
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import load_model
from tensorflow.keras.applications import Xception
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve,make_scorer, accuracy_score, precision_score, recall_score,precision_recall_fscore_support

sgd = tf.keras.optimizers.SGD(learning_rate=1e-3) #宣告 optimizer ， learning rate 設 0.001
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

train_datagen = ImageDataGenerator(rescale=1./255)


test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('CIFAR-10/train', #從資料夾讀出檔案
                                      target_size=(224,224),
                                      batch_size=32)
test_generator = test_datagen.flow_from_directory('CIFAR-10/test',
                                      target_size=(224,224),
                                      batch_size=32,shuffle=False)
def build_model():#建 model
    model =  Xception(include_top=False,  #呼叫 Xception 模型
                 weights='imagenet',
                 input_tensor=Input(shape=(224,224, 3))
                 )
    x = model.output
    x = GlobalAveragePooling2D()(x) #把 Xception 的 output 接到 global average pool 上
    x = Dense(256,activation='relu')(x) # 接上 dense layer 和 activation function ReLU
    predictions = Dense(10, activation='softmax')(x)  # 接上 dense layer 和 activation function ReLU
    model = Model(inputs=model.input, outputs=predictions)
    return model
def train():
    model = build_model()
    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',   #編譯 model 設置 Loss function 、 metric 和 optimizer
              metrics=['accuracy'])
    callback=keras.callbacks.ModelCheckpoint("cnn.h5", monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto') #寫 call back 目的是保存 val accuracy 最佳的 MODEL
    #model.fit_generator(train_generator,epochs=30,callbacks=[callback],validation_data=test_generator) # train 
    model = load_model("cnn.h5") #load 最佳 model
    model.evaluate(test_generator) #測 test accuracy 和 loss
    '''
    y_pred = np.argmax(model.predict(test_generator),axis=1)# model.predict() 出來的數值是各種 class 的機率，故需要用 np.argmax() 做進一步的轉換
    print(y_pred)
    print(loss)
    print(acc)
    print()
    '''
if __name__=='__main__':
    train()

