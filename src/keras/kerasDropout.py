#Optional OS settings
import os
os.environ['TF_XLA_FLAGS'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#Load in relevant libraries
import numpy as np
from numpy import loadtxt, expand_dims
import matplotlib
import tensorflow as tf
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

#Select relevant datasets

# train_dataset = loadtxt('../datasets/conv_training_reduced.csv',delimiter=',')
# val_dataset = loadtxt('../datasets/conv_validation_reduced.csv',delimiter=',')
# test_dataset = loadtxt('../datasets/conv_testing_reduced.csv',delimiter=',')

train_dataset = loadtxt('../datasets/reduced_datasets/training_reduced.csv',delimiter=',')
val_dataset = loadtxt('../datasets/reduced_datasets/validation_reduced.csv',delimiter=',')
test_dataset = loadtxt('../datasets/reduced_datasets/testing_reduced.csv',delimiter=',')

# train_dataset = loadtxt('../datasets/full_datasets/training_full.csv',delimiter=',')
# val_dataset = loadtxt('../datasets/full_datasets/validation_full.csv',delimiter=',')
# test_dataset = loadtxt('../datasets/full_datasets/testing_full.csv',delimiter=',')

# train_dataset = loadtxt('../datasets/full_datasets_npa/training_full_npa.csv',delimiter=',')
# val_dataset = loadtxt('../datasets/full_datasets_npa/validation_full_npa.csv',delimiter=',')
# test_dataset = loadtxt('../datasets/full_datasets_npa/testing_full_npa.csv',delimiter=',')

X = train_dataset[:,0:len(train_dataset[0])-1]
y = train_dataset[:,len(train_dataset[0])-1:]

X_val = val_dataset[:,0:len(val_dataset[0])-1]
y_val = val_dataset[:,len(val_dataset[0])-1:]

X_test = test_dataset[:,0:len(test_dataset[0])-1]
y_test = test_dataset[:,len(test_dataset[0])-1:]

#Reshaping for Conv1D input layer, leave commented if not using CNN

# X = expand_dims(X,axis=2)
# X_val = expand_dims(X_val,axis=2)
# X_test = expand_dims(X_test,axis=2)


dr = 0.5

from tensorflow.keras.regularizers import l1, l2, L1L2

L1 = 0.1
L2 = 0.1

def return_hos_rmse(e):
    return e[2]

test_vals = []

#Print model name and details for record keeping purposes
print('Model Name: DNN2 batch128')
print('Dataset: Reduced')
print('Training Method: Dropout')
print('Batch Normalization: Yes')
print('')

results = []

min_rmse = 1000

#Train 50 different iterations of the network, charting the RMSE for each iteration
#Saving the best network based on testing dataset performance
for i in range(50):
    #DNN-2
    #'''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=19))#, kernel_regularizer=l1(L1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(units=32))#, kernel_regularizer=l1(L1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(units=32))#, kernel_regularizer=l1(L1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    #'''

    #Large DNN
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=80, kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=100, kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=50, kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=25, kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=10, kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    '''

    #CNN
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=80, kernel_size=16, strides=16,input_shape=(80,1),kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.GRU(100, return_sequences='false', kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=50, kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=25, kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=10, kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    '''



    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=[tf.keras.metrics.RootMeanSquaredError()])


    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=0,patience=1)
    history = model.fit(X,y,epochs=500,batch_size=128,verbose=0,validation_data=(X_val,y_val))#,callbacks=[callback])

    if i==0:
        print(model.summary())
        print('')
    #model.save('test_DNN.h5')

    train_mse, train_rmse = model.evaluate(X, y, batch_size=128,verbose=0)
    val_mse, val_rmse = model.evaluate(X_val, y_val, batch_size=128,verbose=0)
    test_mse, test_rmse = model.evaluate(X_test, y_test, batch_size=128,verbose=0)

    print(train_rmse,val_rmse,test_rmse)
    results.append((train_rmse,val_rmse,test_rmse))

    if test_rmse<min_rmse:
        model.save('DNN2_Reduced_Dropout_batch128_BN_Model.h5')

results.sort(reverse=False,key=return_hos_rmse)

np.savetxt('DNN2_Reduced_Dropout_batch128_BN_results.csv', np.asarray(results),fmt='%s', delimiter=',')
