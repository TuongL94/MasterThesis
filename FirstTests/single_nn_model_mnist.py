# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 8:55:36 2018

@author: Tuong Lam & Simon Nilsson
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def inference(input):
    input_layer = input
    
    # Convolutional layer 1
    conv1 = tf.layers.conv2d(
            inputs = input_layer,
            filters = 32,
            kernel_size = [5, 5], 
            padding = "same",
            reuse = tf.AUTO_REUSE,
            activation = tf.nn.relu,
            name = "conv_layer_1")
    
    # Pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, 
                                     pool_size = [2,2], 
                                     strides = 2)
    
    # Convolutional Layer 2 and pooling layer 2
    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 64,
            kernel_size = [5,5],
            padding = "same",
            reuse = tf.AUTO_REUSE,
            activation = tf.nn.relu,
            name = "conv_layer_2")
            
    pool2 = tf.layers.max_pooling2d(
            inputs = conv2, 
            pool_size = [2,2],
            strides = 2)
    
    net = tf.layers.flatten(pool2)
    
    net = tf.layers.dense(
            inputs = net,
            units = 1024,
            reuse =tf.AUTO_REUSE,
            activation = tf.nn.relu,
            name = "dense_layer_1")
    
    net = tf.layers.dropout(
            inputs = net, 
            rate = 0.4)
    
    # Logits Layer
    net = tf.layers.dense(inputs = net, 
                          units = 10,
                          reuse = tf.AUTO_REUSE,
                          name = "dense_layer_2")
    
    return net
    
def training(loss, learning_rate, momentum):
#    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum, use_nesterov=True)
    train_op = optimizer.minimize(loss,global_step = tf.train.get_global_step())
    return train_op
    
def placeholder_inputs(dims,nbr_of_eval_pairs):
    data = tf.placeholder(tf.float32, dims, name="data")
    label = tf.placeholder(tf.float32, [dims[0]], name="label") # 1 if same, 0 if different
    eval_data = tf.placeholder(tf.float32, [nbr_of_eval_pairs, dims[1], dims[2], dims[3]], name="data_eval")
    return data,label,eval_data
    
