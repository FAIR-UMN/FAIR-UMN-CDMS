import os
os.environ['TF_XLA_FLAGS'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
from numpy import loadtxt, expand_dims
import matplotlib
import tensorflow as tf
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

from tensorflow.keras.regularizers import l1, l2, L1L2

# train_dataset = loadtxt('conv_training_reduced.csv',delimiter=',')
# val_dataset = loadtxt('conv_validation_reduced.csv',delimiter=',')
# test_dataset = loadtxt('conv_testing_reduced.csv',delimiter=',')

#train_dataset = loadtxt('../datasets/reduced_datasets/training_reduced.csv',delimiter=',')
#val_dataset = loadtxt('../datasets/reduced_datasets/validation_reduced.csv',delimiter=',')
#test_dataset = loadtxt('../datasets/reduced_datasets/testing_reduced.csv',delimiter=',')

train_dataset = loadtxt('../datasets/full_datasets/training_full.csv',delimiter=',')
val_dataset = loadtxt('../datasets/full_datasets/validation_full.csv',delimiter=',')
test_dataset = loadtxt('../datasets/full_datasets/testing_full.csv',delimiter=',')

# train_dataset = loadtxt('full_datasets_npa/training_full_npa.csv',delimiter=',')
# val_dataset = loadtxt('full_datasets_npa/validation_full_npa.csv',delimiter=',')
# test_dataset = loadtxt('full_datasets_npa/testing_full_npa.csv',delimiter=',')

X = train_dataset[:,0:len(train_dataset[0])-1]
y = train_dataset[:,len(train_dataset[0])-1:]

X_val = val_dataset[:,0:len(val_dataset[0])-1]
y_val = val_dataset[:,len(val_dataset[0])-1:]

X_test = test_dataset[:,0:len(test_dataset[0])-1]
y_test = test_dataset[:,len(test_dataset[0])-1:]

# X = expand_dims(X,axis=2)
# X_val = expand_dims(X_val,axis=2)
# X_test = expand_dims(X_test,axis=2)
#dr = 0.2


L1 = 0.1
L2 = 0.1

EPOCHS = 1
BATCH_SIZE = 50
INPUT_DIM = 85

def return_hos_rmse(e):
    return e[2]

test_vals = []

print('Model Name: CNN')
print('Dataset: FullNPA')
print('Training Method: L1 Reg+Early Stopping')
print('Batch Normalization: No')
print('')

results = []

min_rmse = 1000

for i in range(EPOCHS):
    #DNN-2
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=19, kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=32, kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=32, kernel_regularizer=l1(L1)))
    #model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    '''

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
    #'''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=INPUT_DIM, kernel_size=16, strides=16,input_shape=(INPUT_DIM,1),kernel_regularizer=l1(L1)))
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
    #'''



    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.RootMeanSquaredError()])


    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=0,patience=1)
    history = model.fit(X,y,epochs=i,batch_size=BATCH_SIZE,verbose=0,validation_data=(X_val,y_val),callbacks=[callback])

    if i==0:
        print(model.summary())
        print('')
    #model.save('test_DNN.h5')

    train_mse, train_rmse = model.evaluate(X, y, batch_size=50,verbose=0)
    val_mse, val_rmse = model.evaluate(X_val, y_val, batch_size=50,verbose=0)
    test_mse, test_rmse = model.evaluate(X_test, y_test, batch_size=50,verbose=0)

    print(train_rmse,val_rmse,test_rmse)
    results.append((train_rmse,val_rmse,test_rmse))

    if test_rmse<min_rmse:
        model.save('CNN_FullNPA_L1RegES_noBN_Model.h5')

results.sort(reverse=False,key=return_hos_rmse)

np.savetxt('CNN_FullNPA_L1RegES_noBN_results.csv', np.asarray(results),fmt='%s', delimiter=',')

