# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:55:36 2018

@author: Tuong Lam
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
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            name="conv_layer_1")
    
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
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            name="conv_layer_2")
            
    pool2 = tf.layers.max_pooling2d(
            inputs = conv2, 
            pool_size = [2,2],
            strides = 2)
    
    net = tf.layers.flatten(pool2)
    return net
    
    
def l2_loss(input_1,input_2):
    return tf.linalg.norm([input_1,input_2])
    
def contrastive_loss(input_1,input_2,label,margin):
    """ Computes the contrastive loss between two vectors
    
    Input:
    inout_1 - first input vector
    input_2 - second input vector
    label - ground truth for similarity between the vectors. 1 if they are similar, 0 otherwise.
    margin - margin for contrastive loss, positive constant
    Returns the contrastive loss between input_1 and input_2 with specified margin.
    """
    d_sq = tf.reduce_sum(tf.pow(input_1-input_2,2),1,keep_dims=True)
    max_sq = tf.square(tf.maximum(margin-d_sq,0))
    return tf.reduce_mean(label*d_sq + (1-label)*max_sq)/2
        

def training(loss, learning_rate, momentum):
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum, use_nesterov=True)
    train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op
    
def placeholder_inputs(batch_size,nbr_of_pairs):
    left = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name="left")
    right = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name="right")
    label = tf.placeholder(tf.float32, [batch_size, 1], name="label") # 1 if same, 0 if different
    left_eval = tf.placeholder(tf.float32, [nbr_of_pairs, 28, 28, 1], name="left_eval")
    right_eval = tf.placeholder(tf.float32, [nbr_of_pairs, 28, 28, 1], name="right_eval")
    return left,right,label,left_eval,right_eval
    