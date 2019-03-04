#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 20:06:41 2019

@author: dawei
"""

from numpy.random import seed
from tensorflow import set_random_seed
import numpy as np
import pandas as pd
import os

import keras as K
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling1D
from keras.optimizers import SGD, Adadelta
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import check_dirs


'''Fix random seed'''
seed(0)
set_random_seed(0)

data_path = './csv_files/'   # Path to load training data
save_model_dir = './well_trained_CNN/' # Path to save/load CNN model
label_path = './code_book/'

#%%

def uint8_to_float32(x):   # Standardrization
    return (np.float32(x) - 128.) / 128.

#%%
    
def cnn_model_fn():
        model = Sequential()
        model.add(Conv1D(64, 8, strides=1, activation='relu', padding="valid", input_shape=(128,1)))
        model.add(BatchNormalization())
        #model.add(Dropout(0.1))   # Don't add dropout in conv
        model.add(Conv1D(128, 8, strides=1, activation='relu', padding="valid", input_shape=(128,1)))
        model.add(BatchNormalization())
        #model.add(Dropout(0.1))
        model.add(Conv1D(128, 4, strides=1, activation='relu', padding="valid", input_shape=(128,1)))
        model.add(BatchNormalization())
        #model.add(Dropout(0.1))
        print(model.output.shape)
        #model.add(Conv1D(128, 4, strides=1, activation='relu', padding="same"))
        #model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(512, activation ='linear')) # Do not add batchnorm in dense, break linear
        model.add(Dropout(0.2))
        model.add(Dense(512, activation ='linear'))   # may Overfit if too complicated
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation ='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
            
        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        loss = categorical_crossentropy
        model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])
        return model
    
#%%
    
'''Save model and architecture'''
def save_model(clf, count):
    clf.save(save_model_dir + str(count) + 'temp.hdf5')   # Save model
    yaml_string = clf.to_yaml()
    with open(save_model_dir + \
              str(count) + 'temp.yaml', 'w') as f:   # Save architecture
        f.write(yaml_string)
    f.close()
    
#%%
    
def reshape(training_data_fit, training_labels):
    # Reshape training data as (#,128,1) for CNN
    training_data_fit = np.reshape(training_data_fit, (training_data_fit.shape[0], 128, 1))   
    # One-hot encoding for training labels: (#,3)
    training_labels = np_utils.to_categorical(training_labels, 3)
    return training_data_fit, training_labels

#%%
if __name__ == "__main__":
    # Create a path to save model
    check_dirs.check_dir(save_model_dir) 
    
    # Load target classes
    classes = pd.read_csv(label_path + 'UCLA_data_codebook.csv').values[:,0]
    classes = classes.tolist()
    for i in range(len(classes)):
        classes[i] = classes[i][0:-5]   # Only reserve T1_(x)s
    classes = list(set(classes))   # Merge same elements in classes 
    print('target class:', classes)
    
    # Load training data
    csv_list = [x for x in os.listdir(data_path) if x.endswith('.csv')]   
    print('Number of csv files:', len(csv_list))
    training_data = np.ones((1,128))
    labels = []   # Training labels
    for i in range(len(csv_list[0:400])):
        for j in range(len(classes)):
            if csv_list[i].endswith(classes[j] + '.csv'):
                print('csv:', i)
                file_dir = os.path.join(data_path, csv_list[i])
                temp_data = pd.read_csv(file_dir, header=None).values
                # Segmentation for every 5 sec
                for length in range(0, temp_data.shape[0], 20):
                    try:
                        segment = temp_data[length:length+20].mean(axis=0)
                        labels.append(j)
                        training_data = np.vstack((training_data, segment))
                    except:
                        pass
                                                         
    # Normalization, otherwise CNN not works!!!
    training_data = uint8_to_float32(training_data[1:]) 
    labels = np.asarray(labels)
    

#%%   
    # define 3-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    cvscores = []
    count = 0
    for train, test in kfold.split(training_data, labels):
        count += 1
        # training
        training_data_fit, training_labels = reshape(training_data[train], labels[train])
        print ('training_data_fit shape:', training_data_fit.shape)
        print ('training_labels shape:', training_labels.shape)
        eval_data, eval_labels = reshape(training_data[test], labels[test])
        print ('eval_data shape:', eval_data.shape)
        print ('eval_labels shape:', eval_labels.shape)
        model = cnn_model_fn()
        model.fit(training_data_fit, training_labels,   
            batch_size=32,
            epochs=100,
            verbose=2,
            validation_data = (eval_data, eval_labels),
            shuffle=True,
            callbacks=[EarlyStopping(monitor='val_acc', patience=5, mode='auto')])
        save_model(model, count)
        print('Well trained and saved')
        
        # evaluate the model
        scores = model.evaluate(eval_data, eval_labels, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
