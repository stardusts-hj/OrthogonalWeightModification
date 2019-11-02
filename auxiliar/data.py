"""
Timo Flesch, 2019
"""
import pickle
import tensorflow as tf
if int(tf.VERSION[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS

def gen_splitMNIST(bounds):
    dataset = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
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
