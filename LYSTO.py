#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
import matplotlib.pyplot as plt

data = h5py.File('breast.h5', 'r')
X, Y, P = data['images'], np.array(data['counts']), np.array(data['id'])


from matplotlib.colors import LinearSegmentedColormap
from skimage.color import rgb2hed

cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white', 'saddlebrown'])

img0 = rgb2hed(X[0])
arr = np.array(img0[:,:,2])
img1 = rgb2hed(X[1])
arr = np.stack((arr, np.array(img1[:,:,2])), axis=0)

for i in range(len(X[2:])):
    img = rgb2hed(X[i+2])
    i = np.array(img[:,:,2])
    arr = np.concatenate([arr, [i]], axis=0)

print(arr.shape)


from numpy import newaxis
from sklearn.model_selection import train_test_split

arr = arr[:,:,:,newaxis]

print(arr.shape)

X_tr, X_val, y_tr, y_val = train_test_split(arr, Y, test_size=0.3)
X_val, X_tt, y_val, y_tt = train_test_split(X_val, y_val, test_size=0.5)

print(X_tr.shape, y_tr.shape)
print(X_val.shape, y_val.shape)
print(X_tt.shape, y_tt.shape)


from keras.models import Sequential
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

# early stopping is used to regulate the amount of training in order to prevent over-fitting
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

model = Sequential()

model.add(Conv2D(16, (3,3), input_shape=(299,299,1))) # layer 1 of convolution network with 16 filters
model.add(MaxPooling2D(pool_size=3))                     # max pooling layer with a matrix of size 3 x 3

model.add(Conv2D(32, (3,3)))                # layer 2 of convolution network with 32 filters
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(64, (3,3)))                # layer 3 of convolution network with 64 filters
model.add(MaxPooling2D(pool_size=3))

model.add(Activation('relu'))
model.add(Flatten())                        # layer to flatten the matrix into vector
model.add(Dense(1))                         # dense layer to output the predicted cell count 

# model compilation with adam optimizer and mean squared error as loss function, 
# r2 score is used as evaluation metric
model.compile(optimizer='rmsprop', loss='mean_absolute_error', metrics=[r_square])

history = model.fit(X_tr, y_tr, batch_size=100, nb_epoch=160, verbose=1, validation_data=(X_val, y_val), callbacks=[earlystopping])


import math 
from scipy.stats import pearsonr

y_pred = model.predict(X_tt)

mse_score = mean_squared_error(y_tt, y_pred)
R2_score = r2_score(y_tt, y_pred)
# corr, _ = pearsonr(Ytest, y_pred)

print("Test Scores")
print("RMSE Loss: ", math.sqrt(mse_score))
print("R2 Score: ", R2_score)
# print("Correlation Coefficient: ", corr)

print("Scatter plot of predicted value vs. true value")
plt.scatter(y_tt, y_pred)
plt.xlabel('True value')
plt.ylabel('Predicted value')
plt.show()
