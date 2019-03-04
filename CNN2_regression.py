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
import csv
import os

import keras as K
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling1D
from keras.optimizers import SGD, Adadelta
from keras.losses import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import check_dirs

#%%

'''Fix random seed'''
seed(0)
set_random_seed(0)

data_path = './csv_files/'   # Path to load training data
save_model_dir = './well_trained_CNN/' # Path to save/load CNN model
label_path = './code_book/'
prediction_path = './prediction/'
os.chdir('/home/dawei/research3')   # Set main dir
print(os.path.abspath(os.path.curdir))

mode = 'training'   # Either 'training' or 'test'
target = 'WSS'
t = 10

#%%

def uint8_to_float32(x):   # Standardrization
    return (np.float32(x) - 128.) / 128.

#%%
    
def normalize_label(y):
    sc = StandardScaler()
    return sc.fit_transform(y)

#%%
    
def cnn_model_fn(num_of_classes):
        model = Sequential()
        # Block 1
        model.add(Conv1D(64, 8, strides=1, activation='relu', kernel_initializer= 'normal', padding="same", input_shape=(128,1)))
        #model.add(Dropout(0.1))   # Don't add dropout in conv
        model.add(Conv1D(64, 8, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        # Block 2
        #model.add(Conv1D(128, 8, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        #model.add(Conv1D(128, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))


        #model.add(Conv1D(256, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        #model.add(Conv1D(256, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        #model.add(Conv1D(256, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        
        #model.add(Conv1D(512, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        #model.add(Conv1D(512, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        #model.add(Conv1D(512, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        print(model.output.shape)

        model.add(Flatten())
        model.add(Dense(512, activation ='relu', kernel_initializer= 'normal')) # Do not add batchnorm in dense, break linear
        model.add(Dropout(0.2))
        model.add(Dense(512, activation ='relu', kernel_initializer= 'normal'))   # may Overfit if too complicated
        model.add(Dropout(0.2))
        #model.add(Dense(1024, activation ='linear', kernel_initializer= 'normal'))
        #model.add(Dropout(0.2))
        model.add(Dense(num_of_classes, kernel_initializer= 'normal'))
            
        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        loss = mean_squared_error
        model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['mae'])
        return model
    
#%%
    
'''Save model and architecture'''
def save_model(clf, name):
    clf.save(save_model_dir + name + '.hdf5')   # Save model
    yaml_string = clf.to_yaml()
    with open(save_model_dir + \
              name + '.yaml', 'w') as f:   # Save architecture
        f.write(yaml_string)
    f.close()
    
#%%
    
def reshape(training_data_fit):
    # Reshape training data as (#,128,1) for CNN
    training_data_fit = np.reshape(training_data_fit, (training_data_fit.shape[0], 128, 1))   
    
    return training_data_fit

#%%
if __name__ == "__main__":
    # Create a path to save model
    check_dirs.check_dir(save_model_dir) 
    
    # Load target classes
    classes = pd.read_csv(label_path + 'UCLA_data_codebook.csv').values[:,0]
    classes = classes.tolist()
    class_to_select = [e for e in classes if target in e]
    print('target class:', class_to_select)

    # Load regression labels (values)
    UCLA_data = pd.read_csv(label_path + 'UCLA_data.csv')
    # Replace empty str as null, then drop
    UCLA_data = UCLA_data.replace(' ', np.nan, inplace=False) 
    UCLA_data = UCLA_data.dropna(how='any')   
    # File idx, int
    csid = UCLA_data.values[:,0].tolist()   
    # Labels, float
    reg_label = UCLA_data[class_to_select].values.astype(float)   
  
    # Load training data
    csv_list = [x for x in os.listdir(data_path) if x.endswith('.csv')]   
    print('Number of csv files:', len(csv_list))
    training_data = np.ones((1,128))
    labels = []   # Training labels
    file_marker = []
    # Match csid + target class with file names
    for i in range(len(csid[0:])):
        for j in range(len(csv_list)):
            if csv_list[j].startswith(str(csid[i])) and target in csv_list[j]:
                print('%d csid: %d' %(i, csid[i]))
                file_dir = os.path.join(data_path, csv_list[j])
                temp_data = pd.read_csv(file_dir, header=None).values
                # Segmentation for every 20 sec
                for length in range(0, temp_data.shape[0], t):
                    try:
                        segment = temp_data[length:length+t].mean(axis=0)
                        labels.append(reg_label[i])
                        training_data = np.vstack((training_data, segment))
                        file_marker.append(csid[i])
                    except:
                        pass
    
    labels = np.asarray(labels)   # List of arrays => array
    training_data= reshape(training_data)   # Reshape for CNN                                                  
    # Normalization, otherwise CNN not works!!!
    training_data = uint8_to_float32(training_data[1:]) 
    #labels = normalize_label(labels)   # Not needed in our model
    
    

#%%   
    # define 3-fold cross validation test harness
    if mode == 'training':
        kfold = KFold(n_splits=3, shuffle=True, random_state=0)
        cvscores = []
        count = 0
        for train, test in kfold.split(training_data, labels):
            # training
            training_data_fit, training_labels = training_data[train], labels[train]
            print ('training_data_fit shape:', training_data_fit.shape)
            print ('training_labels shape:', training_labels.shape)
            eval_data, eval_labels = training_data[test], labels[test]
            print ('eval_data shape:', eval_data.shape)
            print ('eval_labels shape:', eval_labels.shape)
            
            num_of_classes = len(class_to_select)
            model = cnn_model_fn(num_of_classes)
            model.fit(training_data_fit, training_labels,   
                batch_size=32,
                epochs=100,
                verbose=2,
                validation_data = (eval_data, eval_labels),
                shuffle=True,
                callbacks=[EarlyStopping(monitor='val_mean_absolute_error', 
                                         patience=5, 
                                         mode='auto')])
            
            # evaluate the model
            scores = model.evaluate(eval_data, eval_labels, verbose=1)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
            # Save hdf5 and yaml
            save_model(model, name = target + 
                                      '-reg' + str(count) +
                                      '-%.4f' %scores[1])
            count += 1
            print('Well trained and saved')
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    elif mode == 'test':
        num_of_classes = len(class_to_select)
        model = cnn_model_fn(num_of_classes)
        print('Predicted class:', target)
        model.load_weights(save_model_dir + 'HSS-reg0-0.4735.hdf5')
        pred = model.predict(training_data, batch_size=32)
        
        with open(prediction_path + target + '_pred.csv', mode='w') as file:
            file_writer = csv.writer(file, delimiter=',')
            print(pred.shape)
            for i in range(len(pred)):
                file_writer.writerow([file_marker[i], pred[i]])
        print('prediction saved')
        