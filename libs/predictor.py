# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.ops.rnn import dynamic_rnn

from .bnlstm import LSTMCell, BNLSTMCell, orthogonal_initializer

# Weights and Bias initializer
# —————————————————————————————————————————————————————————————————————————————
def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial, name=name)

# Batch Normalization
# —————————————————————————————————————————————————————————————————————————————
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

def batch_norm_layer(x,train_phase,scope_bn='bn'):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
                          updates_collections=None,
                          is_training=True,
                          reuse=None, # is this right?
                          trainable=True,
                          scope=scope_bn)
    bn_test = batch_norm(x, decay=0.999, center=True, scale=True,
                         updates_collections=None,
                         is_training=False,
                         reuse=True, # is this right?
                         trainable=True,
                         scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_test)
    return z

#==============================================================================
#%% 1-D CNN
class CNN_x:
    def __init__(self, x, y, problem, n_input, n_output,
                 keep_prob, BN=False, training = None, DFS=True):
        X = tf.reshape(x, [-1, n_input, 1])
        self. conv1 = tf.layers.conv1d(
            inputs = X,
            filters = 16,
            kernel_size = 1000,
            padding = 'same',
            activation = tf.nn.relu,
            kernel_initializer = tf.truncated_normal_initializer,
            bias_initializer = tf.truncated_normal_initializer,
            name = 'conv1'
        )

        pool1 = tf.layers.max_pooling1d(
            inputs = self.conv1,
            pool_size = 2000,
            strides = 2000
        )

        # important!!!
        #############################################
        # n_input/kernel_size
        pool2_flat = tf.reshape(pool1, [-1, int(n_input/2000) * 16])
        # pool2_flat = tf.reshape(pool2, [-1, 50 * 64])
        #############################################

        # full connection
        dense = tf.layers.dense(
            inputs = pool2_flat,
            units = 32
        )

        pool2_drop = tf.nn.dropout(dense, keep_prob)

        # if problem == "classification":
        #     # full connection
        #     fclayer = tf.layers.dense(
        #         inputs=pool2_drop,
        #         units=2
        #     )
        #     self.pred = tf.nn.softmax(fclayer)
        #     y = tf.concat([y, 1-y], 1)
        #     self.loss = tf.reduce_mean(tf.nn. softmax_cross_entropy_with_logits(logits=fclayer, labels=y))
        #     self.yres = y - self.pred
        # if problem == "regression":
            # full connection

        fclayer = tf.layers.dense(
            inputs=pool2_drop,
            units=1
        )
        self.pred = fclayer
        self.loss = tf.nn.l2_loss((tf.subtract(self.pred, y)))
        self.lossSmooth = tf.nn.l2_loss(tf.subtract(self.pred[:-1,:], self.pred[1:,:]))
        self.yres = y - self.pred


# %% null_model
class null_model:
    def __init__(self, x , y, problem = 'none', n_input = 1, n_output = 1,
                 keep_prob = 0.1, BN=False, training = None, DFS=True):
        self.pred = tf.zeros(tf.shape(y))
        self.loss = tf.nn.l2_loss((tf.subtract(y, y)))
        self.lossSmooth = tf.nn.l2_loss((tf.subtract(y, y)))

#%% LSTM

class LSTM:
    def __init__(self, x, y, y_cnn, problem, n_input, n_steps, n_hidden, n_classes,
                 keep_prob):
        # create the variable of feature weights
        self.w = tf.Variable(np.ones([1, n_input]), dtype=tf.float32, name="DFS_weights")
        self.wpenalty1 = tf.reduce_sum(tf.abs(self.w))
        self.fbatch = tf.Variable(np.ones([1, n_input]), dtype=tf.float32, trainable=False)
        # weight the input
        x_weighted = tf.reshape(tf.multiply(x, tf.multiply(self.w, self.fbatch)), [-1, n_steps, n_input])
        x_weighted = tf.nn.dropout(x_weighted, keep_prob)

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x_ = tf.unstack(x_weighted, n_steps, 1)
        
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        
        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x_, dtype=tf.float32)

        # Loss
        if problem == "classification":
            # Define weights
            self.weights = {'out': weight_variable([n_hidden, 2], name='rnn_w_out')}
            self.biases = {'out': bias_variable([2], name='rnn_b_out')}
            # Linear activation, using rnn inner loop last output
            y = tf.concat([y, 1 - y], 1)
            self.pred = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
            self.bpred = tf.nn.softmax(self.pred)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred + y_cnn, labels=y))
        if problem == "regression":
            # Define weights
            self.weights = {'out': weight_variable([n_hidden, n_classes], name='rnn_w_out')}
            self.biases = {'out': bias_variable([n_classes], name='rnn_b_out')}
            # Linear activation, using rnn inner loop last output
            self.pred = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
            self.loss = tf.nn.l2_loss((tf.subtract(self.pred, y-y_cnn)))
