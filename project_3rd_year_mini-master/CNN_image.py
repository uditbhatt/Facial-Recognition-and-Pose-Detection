# -*- coding: utf-8 -*-
"""
Spyder Editor

manish luthyagi
"""

# importing the libraries
from keras.models import Sequential as sequ
from keras.layers import Convolution2D as conv
from keras.layers import MaxPool2D as maxp
from keras.layers import Flatten as Fltn
from keras.layers import Dense as Dns
from keras.layers.core import Dropout as drp
from keras.layers.core import Activation as actv
from keras.layers.normalization import BatchNormalization as btchnrm

import pickle as pckl


#initialize the cnn
model_cnn = sequ()


# convolution
model_cnn.add(conv(32, (3, 3), padding="same", input_shape=(64,64,3)))
model_cnn.add(actv("relu"))
model_cnn.add(btchnrm(axis= -1))
model_cnn.add(maxp(pool_size=(3, 3)))
model_cnn.add(drp(0.25))

model_cnn.add(conv(64, (3, 3), padding="same"))
model_cnn.add(actv("relu"))
model_cnn.add(btchnrm(axis= -1 ))
model_cnn.add(conv(64, (3, 3), padding="same"))
model_cnn.add(actv("relu"))
model_cnn.add(btchnrm(axis= -1))
model_cnn.add(maxp(pool_size=(2, 2)))
model_cnn.add(drp(0.25))

 # pooling
model_cnn.add(maxp(pool_size = (2,2)))

 # Flattening
model_cnn.add(Fltn())

 # Full connection
model_cnn.add(Dns(output_dim = 128, actv = 'relu'))
model_cnn.add(Dns(output_dim = 1, actv = 'sigmoid'))

#compiling the cnn
model_cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#part2- fitting the cnn to the imgs

from keras.preprocessing.img import imgDataGenerator as imgDG

train_datagen_set = imgDG( rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen_set = imgDG(rescale=1./255)

training_data = train_datagen_set.flow_from_directory('train' ,target_size=(64,64),batch_size=32,class_mode='binary')
test_data = test_datagen_set.flow_from_directory('test', target_size=(64,64),batch_size=32,class_mode='binary')

model_cnn.fit_generator(training_data, steps_per_epoch = 80, epochs = 10, validation_data= test_data, validation_steps= 800)

# Dont run
file_model = open('new_model_cnn.pickle','wb')
file_model.write(pckl.dumps(model_cnn))
file_model.close()





