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
            filters = 10,
            kernel_size = [3,3],
            padding = "same",
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            name="conv_layer_2")
            
    pool2 = tf.layers.max_pooling2d(
            inputs = conv2, 
            pool_size = [2,2],
            strides = 2)
    
    net = tf.layers.flatten(pool2)
    
    net = tf.layers.dense(
            inputs = net,
            units = 1024,
            reuse = tf.AUTO_REUSE,
            activation = tf.nn.relu)
    net = tf.layers.dropout(
            inputs = net, 
            rate = 0.4)
    
    # Logits Layer
    net = tf.layers.dense(inputs = net, units = 10)
    
    return net
    
    
def l2_loss(input_1,input_2):
    return tf.linalg.norm([input_1,input_2])
    
def contrastive_loss(input_1,input_2,label,margin):
    """ Computes the contrastive loss between two vectors
    
    Input:
    input_1 - first input vector
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
    
def placeholder_inputs(dims,nbr_of_eval_pairs):
    """ Creates placeholders for siamese neural network.
    
    This method returns placeholders for inputs to the siamese network in both
    training and evaluation, it also returns a placeholder for ground truth of 
    image pairs.
    Input:
    dims - list of following structure: [batch_size height width 1] (the 1 is for grayscale images)
    nbr_of_eval_pairs - number of image pairs to use for evaluation
    Returns:
    left - placeholder for left input to siamese network for training
    right - placeholder for right input to siamese network for training
    label - placeholder for ground truths of image pairs for training
    left_eval - placeholder for left input to siamese network for evaluation
    right_eval - placeholder for right input to siamese network for evaluation
    """
    data = tf.placeholder(tf.float32, dims, name="data")
    label = tf.placeholder(tf.float32, [dims[0], 1], name="label") # 1 if same, 0 if different
    eval_data = tf.placeholder(tf.float32, [nbr_of_eval_pairs, dims[1], dims[2], dims[3]], name="data_eval")
    return data,label,eval_data
    