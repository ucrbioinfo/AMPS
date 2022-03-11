from tensorflow import keras
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.constraints import max_norm
import numpy as np
import pandas as pd


import preprocess as preprocess
import profile_generator as pg
import random
from datetime import datetime
import tensorflow as tf

def test_sampler(methylations_test, sequences_onehot, annot_seqs_onehot, window_size, num_to_chr_dic, include_annot=False):
    methylated, unmethylated = preprocess.methylations_subseter(methylations_test, window_size)
    test_sample_size = int(min(50000, 2*len(methylated), 2*len(unmethylated)))
    test_sample_set = methylated[:test_sample_size]+unmethylated[:test_sample_size]
    random.shuffle(test_sample_set)
    test_profiles, test_targets = pg.get_profiles(methylations_test, test_sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=window_size)
    x_test, y_test = preprocess.cpgenie_preprocess(test_profiles, test_targets)
    return x_test, y_test

def cpgenie_preprocess(X, Y):
    X = np.delete(X, range(4, X.shape[2]), 2)
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    b = np.zeros((Y.size, Y.max()+1))
    b[np.arange(Y.size), Y] = 1
    Y = b
    X = X.reshape(list(X.shape) + [1])
    X = np.swapaxes(X, 1, 2)
    return X, Y

def run_experiments(methylations, sequences_onehot, num_to_chr_dic, data_size, memory_chunk_size=10000):
    methylations_train, methylations_test = preprocess.seperate_methylations('', methylations, from_file=False)
    PROFILE_ROWS = 1000
    PROFILE_COLS = 4

    W_maxnorm = 3
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(1, 5), activation='relu', input_shape=(PROFILE_COLS, PROFILE_ROWS, 1), padding='same', kernel_constraint=max_norm(W_maxnorm)))
    model.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 3)))
    model.add(Conv2D(256, kernel_size=(1, 5), activation='relu', padding='same', kernel_constraint=max_norm(W_maxnorm)))
    model.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 3)))
    model.add(Conv2D(512, kernel_size=(1, 5), activation='relu', padding='same', kernel_constraint=max_norm(W_maxnorm)))
    model.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 3)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    myoptimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss='binary_crossentropy', optimizer=myoptimizer, metrics=['accuracy'])

    methylated_train, unmethylated_train = preprocess.methylations_subseter(methylations_train, 1000)
    ds_size = min(len(methylated_train), len(unmethylated_train))
    x_train_sz = 0
    step = data_size
    print('##################################', step)
    print('#################################', ds_size)
    if ds_size * 2 < data_size:
        step = (ds_size * 2) - 2
    slice = 0
    for chunk in range(slice, slice+int(step/2), memory_chunk_size):
        if chunk+memory_chunk_size > slice+int(step/2):
            sample_set = methylated_train[chunk:slice+int(step/2)]+unmethylated_train[chunk:slice+int(step/2)]
        else:
            sample_set = methylated_train[chunk:chunk+memory_chunk_size]+unmethylated_train[chunk:chunk+memory_chunk_size]
        random.shuffle(sample_set)
        profiles, targets = pg.get_profiles(methylations_train, sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=1000)
        X, Y = cpgenie_preprocess(profiles, targets)
        x_train, x_val, y_train, y_val = pg.split_data(X, Y, pcnt=0.1)
        x_train_sz += len(x_train)
        with tf.device('/device:GPU:0'):
            print('model fitting started')
            model.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1, validation_data=(x_val, y_val))
            print('model fitting ended')
            print(datetime.now())
            del x_train, y_train
    model.save('./models/' + 'cpgenie_tested_model.mdl')

    x_test, y_test = test_sampler(methylations_test, sequences_onehot, annot_seqs_onehot, 1000, num_to_chr_dic, include_annot=include_annot)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print('Accuracy is : '+str(accuracy_score(y_test, y_pred.round())))



