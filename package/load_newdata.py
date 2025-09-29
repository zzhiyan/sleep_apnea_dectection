import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import os
import random as rn
from numpy.random import seed
import random
random_seed = 7
random.seed(random_seed)
seed(random_seed)
os.environ['PYTHONHASHSEED'] = '0'
rn.seed(12345)


#---------------------------------------------------------------------------------------------
def load_data1(train_data, val_data, test_data):
    print('loading data ...')
    train, SA_train = encode(train_data)
    val, SA_val = encode(val_data)
    test, SA_test = encode(test_data)

    train_mean = np.mean(train)
    train_std = np.std(train)
    X_train = (train - train_mean) / train_std
    X_val = (val - train_mean) / train_std
    X_test = (test - train_mean) / train_std

    X_train = X_train.reshape(X_train.shape[0], 500, 1)
    X_valid = X_val.reshape(X_val.shape[0], 500, 1)
    X_test = X_test.reshape(X_test.shape[0], 500, 1)

    SA_train = np_utils.to_categorical(SA_train, 2)
    SA_val = np_utils.to_categorical(SA_val, 2)
    SA_test = np_utils.to_categorical(SA_test, 2)
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    return X_train, X_valid, X_test, SA_train, SA_val, SA_test,train_mean,train_std

def encode(train):
    train_data = train['RRdata']
    train_data = list(map(eval, train_data))
    train_data = np.array(train_data[:])
    label = LabelEncoder().fit(train.label_SA)
    label_sa = label.transform(train.label_SA)
    return train_data, label_sa


#---------------------------------------------------------------------------------------------
def load_data2(train_data, val_data, test_data):

    train, SA_train = encode(train_data)
    val, SA_val = encode(val_data)
    test, SA_test = encode(test_data)
    entropy, entropy_sum, sort  = train_data['entropy'], train_data['entropy_sum'], train_data['sort']
    entropy = list(map(eval, entropy))
    entropy = np.array(entropy[:])
    entropy_sum = list(map(eval, entropy_sum))
    entropy_sum = np.array(entropy_sum[:])
    sort = list(map(eval, sort))
    sort = np.array(sort[:])

    train_mean = np.mean(train)
    train_std = np.std(train)
    X_train = (train - train_mean) / train_std
    X_val = (val - train_mean) / train_std
    X_test = (test - train_mean) / train_std

    X_train = X_train.reshape(X_train.shape[0], 500, 1)
    X_valid = X_val.reshape(X_val.shape[0],  500, 1)
    X_test = X_test.reshape(X_test.shape[0], 500, 1)
    SA_train = np_utils.to_categorical(SA_train, 2)
    SA_val = np_utils.to_categorical(SA_val, 2)
    SA_test = np_utils.to_categorical(SA_test, 2)
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    return X_train, X_valid, X_test, SA_train, SA_val, SA_test,  entropy, entropy_sum, sort,train_mean,train_std

def load_data3(train_data, val_data, test_data):

    train, SA_train = encode(train_data)
    val, SA_val = encode(val_data)
    test, SA_test, labels_4 = encode3(test_data)

    train_mean = np.mean(train)
    train_std = np.std(train)
    X_train = (train - train_mean) / train_std
    X_val = (val - train_mean) / train_std
    X_test = (test - train_mean) / train_std

    X_train = X_train.reshape(X_train.shape[0], 500, 1)
    X_valid = X_val.reshape(X_val.shape[0],  500, 1)
    X_test = X_test.reshape(X_test.shape[0], 500, 1)
    SA_train = np_utils.to_categorical(SA_train, 2)
    SA_val = np_utils.to_categorical(SA_val, 2)
    SA_test = np_utils.to_categorical(SA_test, 2)
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    return X_train, X_valid, X_test, SA_train, SA_val, SA_test,  train_mean,train_std, labels_4


def load_data500(train_data, val_data, test_data):
    print('loading data ...')
    train, SA_train = encode(train_data)
    val, SA_val = encode(val_data)
    test, SA_test = encode(test_data)

    train_mean = np.mean(train)
    train_std = np.std(train)
    X_train = (train - train_mean) / train_std
    X_val = (val - train_mean) / train_std
    X_test = (test - train_mean) / train_std

    X_train = X_train.reshape(X_train.shape[0], 500, 1)
    X_valid = X_val.reshape(X_val.shape[0], 500, 1)
    X_test = X_test.reshape(X_test.shape[0], 500, 1)

    SA_train = np_utils.to_categorical(SA_train, 2)
    SA_val = np_utils.to_categorical(SA_val, 2)
    SA_test = np_utils.to_categorical(SA_test, 2)
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    return X_train, X_valid, X_test, SA_train, SA_val, SA_test,train_mean,train_std


def encode3(train):
    train_data = train['RRdata']
    train_data = list(map(eval, train_data))
    train_data = np.array(train_data[:])
    label = LabelEncoder().fit(train.label_SA)
    label_sa = label.transform(train.label_SA)

    label = LabelEncoder().fit(train.label_four)
    labels_4 = label.transform(train.label_four)
    return train_data, label_sa, labels_4


