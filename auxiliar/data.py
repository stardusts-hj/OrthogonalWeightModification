import pickle
import tensorflow as tf
import numpy as np
import scipy
import scipy.io as sio
# from scipy.stats import multivariate_normal
from sys import version_info, argv
from datetime import datetime
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes


def gen_splitMNIST(dataset, bounds):

    sets = ["train", "validation", "test"]
    sets_list = []
    for set_name in sets:
        this_set = getattr(dataset, set_name)
        maxlabels = np.argmax(this_set.labels, 1)
        sets_list.append(DataSet(this_set.images[((maxlabels >= bounds[0]) & (maxlabels <= bounds[1])),:],
                                this_set.labels[((maxlabels >= bounds[0]) & (maxlabels <= bounds[1]))],
                                 dtype=dtypes.uint8, reshape=False))
    return base.Datasets(train=sets_list[0], validation=sets_list[1], test=sets_list[2])




def create_customlabels(y):
    labels = np.zeros((y.shape[0],4))
    # odd numbers
    labels[(y % 2) != 0, 0] = 1
    # even numbers
    labels[(y % 2) == 0, 1] = 1
    # small numbers
    labels[y < 5, 2] = 1
    # large numbers
    labels[y > 4, 3] = 1
    return labels


def gen_mbatch(x,y,batch_size=128):
    """ small helper function to retrieve a sample batch """

    sampleStartIDX = np.random.randint(0,len(x)-batch_size)
    x = x[sampleStartIDX:(sampleStartIDX+batch_size),:]
    y = y[sampleStartIDX:(sampleStartIDX+batch_size),:]
    return x, y


def shuff_data(x,y,z):
    """ helper function, shuffles data """
    ii_shuff = np.random.permutation(x.shape[0])
    # shuffle data
    x = x[ii_shuff,:]
    y = y[ii_shuff,:]
    z = z[ii_shuff]
    return x, y, z


def make_taskvectors(x):
    x1 = x
    x2 = x
    x1 = np.concatenate((x1,np.ones((x1.shape[0],1)), np.zeros((x1.shape[0],1))), axis=1)
    x2 = np.concatenate((x2,np.zeros((x2.shape[0],1)), np.ones((x2.shape[0],1))), axis=1)
    return x1, x2



def flatten_datamatrix(mat):
    return mat.reshape(mat.shape[0], mat.shape[1]*mat.shape[2])


def prepare_data():
    # load data
    mnist = tf.keras.datasets.mnist

    # divide data in train and test sets
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    # normalise images
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # flatten vectors
    x_train, x_test = flatten_datamatrix(x_train), flatten_datamatrix(x_test)

    # retain only numbers 1 to 8
    x_train = x_train[(y_train > 0) & (y_train < 9)]
    x_test = x_test[(y_test > 0) & (y_test < 9)]
    y_train = y_train[(y_train > 0) & (y_train < 9)]
    y_test = y_test[(y_test > 0) & (y_test < 9)]

    # create custom labels
    labels_train = create_customlabels(y_train)
    labels_test = create_customlabels(y_test)

    # add columns coding for contexts
    x1_train, x2_train = make_taskvectors(x_train)
    x1_test, x2_test = make_taskvectors(x_test)
    x_train = np.concatenate((x1_train, x2_train), axis=0)
    x_test = np.concatenate((x1_test, x2_test), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)
    y_test = np.concatenate((y_test, y_test), axis=0)
    labels_train = np.concatenate((labels_train, labels_train), axis=0)
    labels_test = np.concatenate((labels_test, labels_test), axis=0)

    x_train, labels_train, y_train = shuff_data(x_train, labels_train, y_train)
    x_test, labels_test, y_test = shuff_data(x_test, labels_test, y_test)
    n_samples = x_train.shape[0]
    x_train = x_train[0:n_samples//2]
    y_train = y_train[0:n_samples//2]
    labels_train = labels_train[0:n_samples//2]

    x_test = x_test[0:n_samples//2]
    y_test = y_test[0:n_samples//2]
    labels_test = labels_test[0:n_samples//2]
    x_train, labels_train, y_train = shuff_data(x_train, labels_train, y_train)
    x_test, labels_test, y_test = shuff_data(x_test, labels_test, y_test)

    labels_train_single = np.empty((labels_train.shape[0],1))
    for ii in range(len(labels_train)):
        if x_train[ii,-1]==0:
            labels_train_single[ii,:] = int(labels_train[ii,0] == 1)
        elif x_train[ii,-1]==1:
            labels_train_single[ii,:] = int(labels_train[ii,2] == 1)

    labels_test_single = np.empty((labels_test.shape[0],1))
    for ii in range(len(labels_test)):
        if x_test[ii,-1]==0:
            labels_test_single[ii,:] = int(labels_test[ii,0] == 1)
        elif x_test[ii,-1]==1:
            labels_test_single[ii,:] = int(labels_test[ii,2] == 1)

    return x_train,x_test,y_train,y_test,labels_train,labels_test, labels_train_single, labels_test_single

x_train,x_test,y_train,y_test,labels_train,labels_test, labels_train_single, labels_test_single = prepare_data()
# -----------------------------------------------------------------------------
