"""
Timo Flesch, 2019
"""
import tensorflow as tf
if int(tf.VERSION[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from sys import argv
# import custom model
from trainer import train_nnet
from nnet import NNet

# import custom helper functions
from auxiliar.data import gen_splitMNIST

FLAGS = tf.app.flags.FLAGS


# directories
tf.app.flags.DEFINE_string('data_dir', './data/', " data directory  ")
tf.app.flags.DEFINE_string('logging_dir', './log/', " log/summary directory ")


# dataset
tf.app.flags.DEFINE_integer('n_inputs', 28*28, " number of inputs ")
tf.app.flags.DEFINE_integer('n_classes', 10, " number of output classes ")



# model

tf.app.flags.DEFINE_string('owm', 'batch', " training procedure (none/batch/task) ")

tf.app.flags.DEFINE_string('nonlinearity', 'relu', " activation function ")
tf.app.flags.DEFINE_integer('dim_hidden', 800," dimensionality of hidden layers ")


# training
tf.app.flags.DEFINE_list('lr', [[0.2]], " learning rate array ")
tf.app.flags.DEFINE_list('alpha', [[0.9, 0.6]], " alpha array (note:[0]*.001**lambda) ")
tf.app.flags.DEFINE_string('optimizer', 'Momentum', " optimisation procedure ")
tf.app.flags.DEFINE_float('momentum', 0.9, " momentum strength ")
tf.app.flags.DEFINE_integer('n_epochs', 20, " number of epochs on entire dataset ")
tf.app.flags.DEFINE_integer('display_step', 100, " step size for log to stdout ")
tf.app.flags.DEFINE_integer('batch_size', 128, " training batch size  ")


def main(argv=None):

    FLAGS = tf.app.flags.FLAGS

    dataset_1 = gen_splitMNIST([0, 4])
    dataset_2 = gen_splitMNIST([5, 9])
    tasks = [dataset_1, dataset_2]
    nnet = NNet()
    train_nnet(tasks, nnet, owm_mode=FLAGS.owm)


if __name__ == '__main__':
    " take care of flags on load "
    tf.compat.v1.app.run()
    # main()
