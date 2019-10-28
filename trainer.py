
import tensorflow as tf
import numpy as np
from datetime import datetime
import os

# import custom helper functions
from auxiliar.data import gen_splitMNIST

FLAGS = tf.app.flags.FLAGS


def train_nnet_basic(tasks, nnet):
    results = dict()
    results['acc_train'] = np.zeros((len(tasks),FLAGS.n_epochs))
    results['loss_train'] = np.zeros((len(tasks),FLAGS.n_epochs))
    results['acc_test'] = np.zeros((len(tasks),len(tasks)))
    results['loss_test'] = np.zeros((len(tasks),len(tasks)))

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
            alph = 0
            for jj in range(FLAGS.n_epochs):

                for kk in range(n_steps_epoch):
                    # print(str(jj))
                    batch_x, batch_y = tasks[ii].train.next_batch(FLAGS.batch_size)
                    # step_alpha = [[FLAGS.alpha[0][0]*0.001**(alph/n_steps_total), FLAGS.alpha[0][1]]]
                    step_alpha = [[FLAGS.alpha[0][0]*0.001**(alph/n_steps_total), FLAGS.alpha[0][1]]]
                    train_dict = {nnet.x_in:batch_x,
                                nnet.y_true:batch_y,
                                nnet.lrates:FLAGS.lr,
                                nnet.alphas:[[[0.9],[0.6]]]} # step_alpha
                    # train network
                    acc_train, loss_train, _ = sess.run([nnet.acc, nnet.loss, nnet.bprop], feed_dict=train_dict)
                    alph +=1

                results['acc_train'][ii, jj] = acc_train
                results['loss_train'][ii, jj] = loss_train
                # evaluate network
                test_dict = {nnet.x_in:tasks[ii].test.images[:],
                            nnet.y_true:tasks[ii].test.labels[:]}
                acc_test, loss_test = sess.run([nnet.acc, nnet.loss],feed_dict=test_dict)
                ep_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("{} Training, Task {}/{}, Epoch {}/{}. Training Loss: {:.4f}, Training Acc: {:.2f}.  Test Loss: {:.4f}, Test Acc: {:.2f}".format(ep_time,ii+1,len(tasks),jj+1, FLAGS.n_epochs, loss_train, acc_train*100,loss_test, acc_test*100))
                sess.run(nnet.bprop_owm, feed_dict={nnet.x_in:batch_x,
                nnet.y_true: batch_y,
                nnet.lrates:FLAGS.lr,
                nnet.alphas:[[[0.9],[0.6]]]})


            # if taskOWM:
            #     # pass
            #     sess.run(nnet.bprop_owm, feed_dict={nnet.x_in:tasks[ii].train.images[:],
            #                                         nnet.y_true: tasks[ii].train.labels[:],
            #                                         nnet.lrates:FLAGS.lr,
            #                                         nnet.alphas:[[[0.9],[0.6]]]})
            #     print('trolololo')

            print('>>>> Performance on previous tasks')
            for jj in range(0,ii+1):
                test_dict = {nnet.x_in:tasks[jj].test.images[:],
                            nnet.y_true:tasks[jj].test.labels[:]}
                acc_test, loss_test = sess.run([nnet.acc, nnet.loss],feed_dict=test_dict)
                results['acc_test'][ii, jj] = acc_test
                results['loss_test'][ii, jj] = loss_test
                # ep_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(">>>> Task {}/{}, Test Loss {:.2f}, Test Accuracy {:.2f}".format(jj+1, len(tasks), loss_test, acc_test*100))

                # if task OWM, update projection matrix
                # TODO run brop_owm on mean input
        return results




def train_nnet_owm(tasks, nnet):
    results = dict()
    results['acc_train'] = np.zeros((len(tasks),FLAGS.n_epochs))
    results['loss_train'] = np.zeros((len(tasks),FLAGS.n_epochs))
    results['acc_test'] = np.zeros((len(tasks),len(tasks)))
    results['loss_test'] = np.zeros((len(tasks),len(tasks)))

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
            alph = 0
            for jj in range(FLAGS.n_epochs):

                for kk in range(n_steps_epoch):
                    batch_x, batch_y = tasks[ii].train.next_batch(FLAGS.batch_size)
                    step_alpha = [[FLAGS.alpha[0][0]*0.001**(alph/n_steps_total), FLAGS.alpha[0][1]]]
                    train_dict = {nnet.x_in:batch_x,
                                nnet.y_true:batch_y,
                                nnet.lrates:FLAGS.lr,
                                nnet.alphas:[[[1.4],[1.4]]]}
                    # train network
                    acc_train, loss_train, _ = sess.run([nnet.acc, nnet.loss, nnet.bprop_owm], feed_dict=train_dict)
                    # sess.run([nnet.update_pmat1,nnet.update_pmat2],feed_dict=train_dict)
                    sess.run([nnet.update_pmat1, nnet.update_pmat2], feed_dict={nnet.x_in:tasks[ii].train.images,
                            nnet.y_true: tasks[ii].train.labels,
                            nnet.lrates:FLAGS.lr,
                            nnet.alphas:[[[0.9],[0.6]]]})
                    alph +=1

                results['acc_train'][ii, jj] = acc_train
                results['loss_train'][ii, jj] = loss_train
                # evaluate network
                test_dict = {nnet.x_in:tasks[ii].test.images[:],
                            nnet.y_true:tasks[ii].test.labels[:]}
                acc_test, loss_test = sess.run([nnet.acc, nnet.loss],feed_dict=test_dict)
                ep_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("{} Training, Task {}/{}, Epoch {}/{}. Training Loss: {:.4f}, Training Acc: {:.2f}.  Test Loss: {:.4f}, Test Acc: {:.2f}".format(ep_time,ii+1,len(tasks),jj+1, FLAGS.n_epochs, loss_train, acc_train*100,loss_test, acc_test*100))




            print('>>>> Performance on previous tasks')
            for jj in range(0,ii+1):
                test_dict = {nnet.x_in:tasks[jj].test.images[:],
                            nnet.y_true:tasks[jj].test.labels[:]}
                acc_test, loss_test = sess.run([nnet.acc, nnet.loss],feed_dict=test_dict)
                results['acc_test'][ii, jj] = acc_test
                results['loss_test'][ii, jj] = loss_test
                # ep_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(">>>> Task {}/{}, Test Loss {:.2f}, Test Accuracy {:.2f}".format(jj+1, len(tasks), loss_test, acc_test*100))

                # if task OWM, update projection matrix
                # TODO run brop_owm on mean input
        return results
