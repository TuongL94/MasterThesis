# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 8:55:36 2018

@author: Tuong Lam & Simon Nilsson
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weight".
    weights = tf.get_variable("weight", kernel_shape,
        trainable = True)
    # Create variable named "bias".
    biases = tf.get_variable("bias", bias_shape,
        trainable = True)
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def inference(input):
    
    # Convolutional layer 1
    with tf.variable_scope("conv1"):
        conv1 = conv_relu(
                input,
                [5,5,1,32],
                [32])
    
    # Pooling layer 1
    pool1 = tf.nn.max_pool(
            value = conv1,
            ksize = [1,2,2,1], 
            strides = [1,2,2,1],
            padding = "SAME")
        
    with tf.variable_scope("conv2"):
        conv2 = conv_relu(
                pool1,
                [5,5,32,64],
                [64])   
        
    pool2 = tf.nn.max_pool(
            value = conv2, 
            ksize= [1,2,2,1],
            strides = [1,2,2,1],
            padding = "SAME")
    
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
    
