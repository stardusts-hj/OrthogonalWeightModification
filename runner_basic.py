import pickle
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime
import scipy
import scipy.io as sio
from sys import version_info, argv
import os

# import custom models
from nnet import NNet_Basic
# import custom helper functions
from auxiliar.data import gen_splitMNIST
# from auxiliar.io import *
# from auxiliar.compute import *

# ignore warning messages (remove clutter from stdout)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


FLAGS = tf.app.flags.FLAGS


# directories
# tf.app.flags.DEFINE_string('data_dir',  './data/',
#                            """ (string) data directory           """)
#
# tf.app.flags.DEFINE_string('ckpt_dir', './checkpoints/',
#                             """ (string) checkpoint directory    """)
# log_dir is already defined by abseil, as of r1.14 (fucking hell...)
tf.app.flags.DEFINE_string('logging_dir',          './log/',
                           """ (string) log/summary directory    """)


# dataset
tf.app.flags.DEFINE_integer('n_inputs', 28*28,
                           """ (int) number of inputs """)

tf.app.flags.DEFINE_integer('n_classes', 10,
                           """ (int) number of output classes """)




# model
tf.app.flags.DEFINE_string('model',                'basic',
                            """ (string)  chosen model          """)


tf.app.flags.DEFINE_string('nonlinearity',       'relu',
                            """ (string)  activation function   """)

tf.app.flags.DEFINE_integer('dim_hidden',             800,
                            """ (int) dimensionality of hidden layers """)



# training
tf.app.flags.DEFINE_list('lr',     [[0.05]],
                            """ (list)   learning rate array             """)

tf.app.flags.DEFINE_list('alpha',     [[0.9, 0.6]],
                            """ (list)  alpha array (note:[0]*.001**lambda)  """)

tf.app.flags.DEFINE_float('L2_lambda',            1e-4,
                            """ (float) strength of regulariser """)

tf.app.flags.DEFINE_string('optimizer',       'Momentum',
                            """ (string)   optimisation procedure     """)

tf.app.flags.DEFINE_float('momentum',     0.9,
                            """ (float) momentum strength   """)

tf.app.flags.DEFINE_integer('n_epochs',   20,
                            """ (int)    number of epochs on entire dataset """)

tf.app.flags.DEFINE_integer('display_step',         100,
                            """(int) step size for log to stdout      """)

tf.app.flags.DEFINE_integer('batch_size',         40,
                            """ (int)     training batch size         """)




def train_nnet(tasks):
    nnet = NNet_Basic()
    init_vars = tf.global_variables_initializer()
    # gpu training: only allocate as much VRAM as required
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # create session and do magic
    with tf.Session(config=config) as sess:
        sess.run(init_vars)
        writer = tf.compat.v1.summary.FileWriter(FLAGS.logging_dir, sess.graph)
        # loop over tasks:
        for ii in range(len(tasks)):
            print("Now training split mnist, task %d" % (ii+1))
            n_data = len(tasks[ii].train.labels[:])
            n_steps_total = n_data*FLAGS.n_epochs//FLAGS.batch_size
            n_steps_epoch = n_data//FLAGS.batch_size
            step = 1
            for jj in range(n_steps_total):
                # print(str(jj))
                param_lambda = jj/n_steps_total
                batch_x, batch_y = tasks[ii].train.next_batch(FLAGS.batch_size)
                step_alpha = [FLAGS.alpha[0][0]*0.001**param_lambda, FLAGS.alpha[0][1]]
                train_dict = {nnet.x_in:batch_x,
                            nnet.y_true:batch_y,
                            nnet.lrates:FLAGS.lr,
                            nnet.alphas:step_alpha}
                # train network
                acc_train, loss_train, _ = sess.run([nnet.acc, nnet.loss, nnet.bprop], feed_dict=train_dict)
                if step % (n_steps_epoch) == 0:
                    # evaluate network
                    test_dict = {nnet.x_in:tasks[ii].test.images[:],
                                nnet.y_true:tasks[ii].test.labels[:]}
                    acc_test, loss_test = sess.run([nnet.acc, nnet.loss],feed_dict=test_dict)
                    ep_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print("{} Training, Task {}/{}, Epoch {}/{}. Training Loss: {:.4f}, Training Acc: {:.2f}.  Test Loss: {:.4f}, Test Acc: {:.2f}".format(ep_time, ii+1,len(tasks),step*FLAGS.n_epochs//n_steps_total+1, FLAGS.n_epochs, loss_train, acc_train*100,loss_test, acc_test*100))

                step +=1
            print('>>>> Performance on previous tasks')
            for jj in range(0,ii+1):
                test_dict = {nnet.x_in:tasks[jj].test.images[:],
                            nnet.y_true:tasks[jj].test.labels[:]}
                acc_test, loss_test = sess.run([nnet.acc, nnet.loss],feed_dict=test_dict)
                # ep_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(">>>> {} Task {}/{}, Test Loss {:.2f}, Test Accuracy {:.2f}".format(jj+1, len(tasks), loss_test, acc_test*100))



def main(argv=None):

    FLAGS = tf.app.flags.FLAGS

    # create datasets for the two tasks
    raw_data = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)
    # dataset_1 = gen_splitMNIST(raw_data, [-1, 5]) # 0 to 4
    # dataset_2 = gen_splitMNIST(raw_data, [4, 10]) # 5 to 9
    # tasks = [dataset_1, dataset_2]
    dataset_1 = gen_splitMNIST(raw_data, [-1, 2]) # 0 to 1
    dataset_2 = gen_splitMNIST(raw_data, [1, 4]) # 2 to 3
    dataset_3 = gen_splitMNIST(raw_data, [3, 6]) # 4 to 5
    dataset_4 = gen_splitMNIST(raw_data, [5, 8]) # 6 to 7
    dataset_5 = gen_splitMNIST(raw_data, [7, 10]) # 8 to 9

    tasks = [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]
    train_nnet(tasks)


if __name__ == '__main__':
    """ take care of flags on load """
    tf.compat.v1.app.run()
