import os
import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


class NNet_Basic(object):
    """simple MLP for mnist"""

    def __init__(self):

        self.initialiser = tf.initializers.glorot_normal()

        with tf.name_scope('placeholders'):
            with tf.name_scope('IO'):
                self.x_in = tf.placeholder(tf.float32, [None, FLAGS.n_inputs], name='x_in')
                self.y_true = tf.placeholder(tf.float32, [None, FLAGS.n_classes], name='y_true')
            with tf.name_scope('hyperparams'):
                self.lrates = tf.placeholder(tf.float32, name='lrates')
                self.alphas = tf.placeholder(tf.float32, name='alpha')

        with tf.name_scope('nnet'):
            self.build_nnet()

        with tf.name_scope('optimiser'):
            self.build_optimiser()

    def build_nnet(self):
        with tf.name_scope('hidden'):
            # add column of ones
            self.x1 = tf.concat([self.x_in, tf.tile(tf.ones([1, 1]), [tf.shape(self.x_in)[0], 1])], 1)
            self.weights_hidden = tf.get_variable("w1", shape=[FLAGS.n_inputs+1, FLAGS.dim_hidden], initializer=self.initialiser)
            self.linearity_hidden = tf.matmul(self.x1, self.weights_hidden, name='lin1')
            self.y_hidden = tf.nn.relu(self.linearity_hidden, name='y_h1')

        with tf.name_scope('output'):
            self.x2 = tf.concat([self.y_hidden, tf.tile(tf.ones([1, 1]), [tf.shape(self.y_hidden)[0], 1])], 1)
            self.weights_out = tf.get_variable("w2", shape=[FLAGS.dim_hidden+1, FLAGS.n_classes], initializer=self.initialiser)
            self.scores = tf.matmul(self.x2, self.weights_out, name='scores')
        return

    def build_optimiser(self):
        with tf.name_scope('loss'):
            xent_vect = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_true)
            self.loss = tf.reduce_mean(xent_vect)

        with tf.name_scope('acc'):
            pred = tf.argmax(self.scores, 1, name="outputs")
            self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(self.y_true, 1)), "float"))

        with tf.name_scope('trainer'):
            # self.optimiser = getattr(tf.compat.v1.train,optimizer+'Optimizer')(learning_rate=self.learning_rate)
            self.optimiser = tf.train.MomentumOptimizer(self.lrates[0][0], momentum=FLAGS.momentum)
            grads = self.optimiser.compute_gradients(self.loss, var_list = [self.weights_hidden, self.weights_out])
            for ii, (g, v) in enumerate(grads):
                # numerical stability: apply gradient clipping
                if g is not None:
                    grads[ii] = (tf.clip_by_norm(g, 10), v)
            self.bprop = self.optimiser.apply_gradients([grads[0], grads[1]])


class NNet_OWM(object):
    """
    MLP with batch owm for orthogonal weight updates
    """
    def __init__(self):

        self.initialiser = tf.initializers.glorot_normal()

        with tf.name_scope('placeholders'):
            with tf.name_scope('IO'):
                self.x_in = tf.placeholder(tf.float32, [None, FLAGS.n_inputs], name='x_in')
                self.y_true = tf.placeholder(tf.float32, [None, FLAGS.n_classes], name='y_true')
            with tf.name_scope('hyperparams'):
                self.lrates = tf.placeholder(tf.float32, name='lrates')
                self.alphas = tf.placeholder(tf.float32, name='alpha')

        with tf.name_scope('nnet'):
            self.build_nnet()

        with tf.name_scope('optimiser'):
            self.build_optimiser()

    def build_nnet(self):
        with tf.name_scope('hidden'):
            with tf.name_scope('forward_pass'):
                self.x1 = tf.concat([self.x_in, tf.tile(tf.ones([1, 1]), [tf.shape(self.x_in)[0], 1])], 1)
                self.weights_hidden = tf.get_variable("w1", shape=[FLAGS.n_inputs+1, FLAGS.dim_hidden], initializer=self.initialiser)
                self.linearity_hidden = tf.matmul(self.x1, self.weights_hidden, name='lin1')
                self.y_hidden = tf.nn.relu(self.linearity_hidden, name='y_h1')
            with tf.name_scope('P_matrix'):
                # p mat init to identity
                self.P1 = tf.Variable(tf.eye(int(FLAGS.n_inputs+1)))
                # mean of inputs
                x_mu = tf.reduce_mean(self.x1, 0 , keep_dims=True)
                # multiply p mat with inputs
                k = tf.matmul(self.P1, tf.transpose(x_mu))
                # compute P update
                self.delta_P1 = tf.divide(tf.matmul(k, tf.transpose(k)), self.alphas[0][0] + tf.matmul(x_mu, k))
                # apply update to P
                self.P1 = tf.assign_sub(self.P1, self.delta_P1)

        with tf.name_scope('output'):
            with tf.name_scope('fprop'):
                self.x2 = tf.concat([self.y_hidden, tf.tile(tf.ones([1, 1]), [tf.shape(self.y_hidden)[0], 1])], 1)
                self.weights_out = tf.get_variable("w2", shape=[FLAGS.dim_hidden+1, FLAGS.n_classes], initializer=self.initialiser)
                self.scores = tf.matmul(self.x2, self.weights_out, name='scores')
            with tf.name_scope('P_matrix'):
                self.P2 = tf.Variable(tf.eye(int(FLAGS.dim_hidden+1)))
                # mean of its inputs
                x_mu = tf.reduce_mean(self.x2, 0 , keep_dims=True)
                # multiply p mat with inputs
                k = tf.matmul(self.P2, tf.transpose(x_mu))
                # compute update term
                self.delta_P2 = tf.divide(tf.matmul(k, tf.transpose(k)), self.alphas[0][1] + tf.matmul(x_mu, k))
                # apply to p2
                self.P2 = tf.assign_sub(self.P2, self.delta_P2)

        return

    def build_optimiser(self):
        with tf.name_scope('loss'):
            xent_vect = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_true)
            self.loss = tf.reduce_mean(xent_vect)

        with tf.name_scope('acc'):
            pred = tf.argmax(self.scores, 1, name="outputs")
            self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(self.y_true, 1)), "float"))

        with tf.name_scope('trainer'):
            # self.optimiser = getattr(tf.compat.v1.train,optimizer+'Optimizer')(learning_rate=self.learning_rate)
            self.optimiser = tf.train.MomentumOptimizer(self.lrates[0][0], momentum=FLAGS.momentum)
            grads = self.optimiser.compute_gradients(self.loss, var_list = [self.weights_hidden, self.weights_out])
            for ii, (g, v) in enumerate(grads):
                # numerical stability: apply gradient clipping
                if g is not None:
                    grads[ii] = (tf.clip_by_norm(g, 10), v)
            with tf.name_scope('OWM'):
                grads_hidden = [self.owm(self.P1, grads[0])]
                grads_output = [self.owm(self.P2, grads[1])]
            self.bprop = self.optimiser.apply_gradients([grads_hidden[0], grads_output[0]])

    def owm(self, P, g_v, lr=1.0):
        g_ = lr * tf.matmul(P, g_v[0])
        return g_, g_v[1]
