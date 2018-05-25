# -*- coding: utf-8 -*-
import argparse
import math
import sys
import time
import copy
import matplotlib.pylab as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
from numpy .random import multivariate_normal, permutation

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', '-e', default=10, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=128,
                    help='learning minibatch size')
parser.add_argument('--train_file_name', '-train_name', type=str, default='./normal.csv',
                    help='the file name of the training data set') 
parser.add_argument('--test_file_name', '-test_name', type=str, default='./abn.csv',
                    help='the file name of the test data set')
parser.add_argument('--window_size', '-ws', type=int, default=400,
                    help='window size')
parser.add_argument('--output_file_name', default='log')
parser.set_defaults(test=False)
                    
args = parser.parse_args()
n_epoch = args.epoch   # number of epochs

outputn=args.output_file_name
train_name=args.train_file_name
test_name=args.test_file_name
D=args.window_size #the size of the window width 
batchsize = args.batchsize   # minibatch size
epoch=args.epoch

# load
x_train_data=np.loadtxt(train_name,delimiter=',')
x_test_data=np.loadtxt(test_name,delimiter=',')

# normalization
x_train_data = x_train_data/(np.std(x_train_data))
x_test_data = x_test_data/(np.std(x_test_data))

#split
x_train_data=x_train_data.reshape([x_train_data.shape[0],x_train_data.size/x_train_data.shape[0]])
x_test_data=x_test_data.reshape([x_test_data.shape[0],x_test_data.size/x_test_data.shape[0]])

Split_train_data=x_train_data.reshape([x_train_data.shape[0]/D,x_train_data.shape[1]*D])
Split_test_data=x_test_data.reshape([x_test_data.shape[0]/D,x_test_data.shape[1]*D])

Split_train_data_x=Split_train_data[0:-1,:]
Split_train_data_y=Split_train_data[1::,:]

Split_test_data_x=Split_test_data[0:-1,:]
Split_test_data_y=Split_test_data[1::,:]

img_rows = D
Split_test_data = Split_test_data.reshape(Split_test_data.shape[0], img_rows, 1 , 1)
Split_train_data = Split_train_data.reshape(Split_train_data.shape[0], img_rows, 1 , 1)
Split_test_data_y = Split_test_data.reshape(Split_test_data.shape[0], img_rows)
Split_train_data_y = Split_train_data.reshape(Split_train_data.shape[0], img_rows)

input_shape = (img_rows, 1, 1)

#Model Net
#size of parameters
batch_size = 64
D=400
dr=0.125 

#Start Neural Network
model = Sequential()
#Fully connected 1st layer
model.add(Flatten(input_shape=input_shape)) #0
model.add(Dense(252)) #1
model.add(Activation('relu')) #2
model.add(Dropout(dr))
#Fully connected 2nd layer
model.add(Dense(int(152))) #4
model.add(Activation('relu')) #5
model.add(Dropout(dr))
#Fully connected 3rd layer
model.add(Dense(int(152))) #7
model.add(Activation('relu')) #8
model.add(Dropout(dr))
#Fully connected 3rd layer
model.add(Dense(int(52))) #7
model.add(Activation('relu')) #8
model.add(Dropout(dr))
#Fully connected 3rd layer
model.add(Dense(int(152))) #7
model.add(Activation('relu')) #8
model.add(Dropout(dr))
#Fully connected 4th layer
model.add(Dense(int(252))) #10
model.add(Activation('relu')) #113
model.add(Dropout(dr))
#Fully connected 4th layer
model.add(Dense(352)) #10
model.add(Activation('relu')) #11
model.add(Dropout(dr))
#Fully connected final(5th layer)
model.add(Dense(img_rows)) #19
model.add(Dropout(dr))

model.compile(loss=keras.losses.mean_squared_error, optimizer='adam')

model.summary()

history = model.fit(Split_train_data, Split_train_data_y,
          batch_size=batch_size,
          epochs=epoch,
          verbose=1,
          validation_data=(Split_train_data, Split_train_data_y))

predict=model.predict(Split_test_data)

measured=Split_test_data_y.reshape(800*400)
predicted=predict.reshape(800*400)

Loss_keras=np.power(measured-predicted,2)
mean_window = 1000
Loss_keras_processed = Loss_keras[0:Loss_keras.size-mean_window]

# smoothen anomaly score
for x in range(Loss_keras.size-mean_window):
    Loss_keras_processed[x] = np.mean(Loss_keras[x:x+mean_window])
#normalize
Loss_keras_processed = Loss_keras_processed/(np.std(Loss_keras_processed))
    
# plot results
fig = plt.figure()
plt.xlabel("sample")
plt.ylabel("anomal score")
plt.plot(Loss_keras_processed, label='32FP results')
plt.legend()
plt.show()

fig2 = plt.figure()
plt.xlabel("sample")
plt.ylabel("value")
plt.plot(predicted[155500:165000], label='predicts')
plt.plot(measured[155500:165000], label='measured')
plt.legend()
plt.show()

model.save("anormaly_FC.h5")
print("Saved model to disk")