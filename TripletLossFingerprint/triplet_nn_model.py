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
            filters = 16,
            kernel_size = [7,7], 
            padding = "same",
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
#                kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=4.0),
            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv_layer_1") 
        
    # Pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, 
                                     pool_size = [2,2], 
                                     strides = 2)
    
    # Convolutional Layer 2 and pooling layer 2
    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 16,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(),
            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv_layer_2")
            
    pool2 = tf.layers.max_pooling2d(
            inputs = conv2, 
            pool_size = [2,2],
            strides = 2)
    
    # Convolutional Layer 3
    conv3 = tf.layers.conv2d(
            inputs = pool2,
            filters = 32,
            kernel_size = [3,3],
#            padding = "same",
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv_layer_3")
            
    pool2 = tf.layers.max_pooling2d(
        inputs = conv3, 
        pool_size = [2,2],
        strides = 2)
    
    # Convolutional Layer 4
    conv4 = tf.layers.conv2d(
            inputs = pool2,
            filters = 64,
            kernel_size = [3,3],
#             padding = "same",
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv_layer_4")
            
    pool4 = tf.layers.max_pooling2d(
            inputs = conv4, 
            pool_size = [2,2],
            strides = 2)
    

    net = tf.layers.flatten(pool4)
    
    net = tf.layers.dense(
            net,
            1000,
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="dense_layer_1")
    
    net = tf.layers.dropout(
            inputs = net,
            rate = 0.4)
     
    return net
    
def contrastive_loss(input_1,input_2,label,margin):
    """ Computes the contrastive loss between two vectors
    
    Input:
    input_1 - first input vector
    input_2 - second input vector
    label - ground truth for similarity between the vectors. 1 if they are similar, 0 otherwise.
    margin - margin for contrastive loss, positive constant
    Returns the contrastive loss between input_1 and input_2 with specified margin.
    """
    d_sq = tf.reduce_sum(tf.pow(input_1-input_2,2),1,keepdims=True)
    max_sq = tf.square(tf.maximum(margin-d_sq,0))
    return tf.reduce_mean(label*d_sq + (1-label)*max_sq)/2

def triplet_loss(anchor, positive, negative, margin):
    """ Computes the contrastive loss between two vectors
    
    Input:
        anchor - output from network when input is the anchor image
        positive - output from network when input is the positve image
        negative - output from network when input is the negative image
        margin - margin for triplet loss, positive constant
    Returns:
        loss - the triplet loss 
    """
    distance_pos = tf.reduce_sum(tf.square(anchor - positive),1)
    distance_neg = tf.reduce_sum(tf.square(anchor - negative),1)
    
    loss = tf.maximum(0., margin + distance_pos - distance_neg)
    loss = tf.reduce_mean(loss)
    return loss

def training(loss, learning_rate, momentum):
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum, use_nesterov=True)
    train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op
    
def placeholder_inputs(image_dims,batch_sizes):
    """ Creates placeholders for triplet neural network.
    
    This method returns placeholders for inputs to the triplet network in both
    training, validation and testing. It also returns a placeholder for ground truth of 
    image pairs for training and validation.
    Input:
    image_dims - list of following structure: [height width 1] (the 1 is for grayscale images)
    batch_sizes - list of batch sizes for training,validation and testing respectively 
    Returns:
    anchor_train - placeholder for anchor input to triplet network for training
    positive_train - placeholder for positive input to triplet network for training
    label_train - placeholder for ground truths of image pairs for training
    anchor_val - placeholder for anchor input to triplet network for validation
    positive_val - placeholder for positive input to triplet network for validation
    label_val - placeholder for ground truths of image pairs for validation
    anchor_test - placeholder for anchor input to triplet network for testing
    positive_test - placeholder for positive input to triplet network for testing
    """
    anchor_train = tf.placeholder(tf.float32, [batch_sizes[0],image_dims[0],image_dims[1],image_dims[2]], name="anchor_train")
    positive_train = tf.placeholder(tf.float32,[batch_sizes[0],image_dims[0],image_dims[1],image_dims[2]], name="positive_train")
    negative_train = tf.placeholder(tf.float32,[batch_sizes[0],image_dims[0],image_dims[1],image_dims[2]], name="negative_train")
#    label_train = tf.placeholder(tf.float32, [batch_sizes[0],1], name="label_train") # 1 if same, 0 if different
    
    anchor_val = tf.placeholder(tf.float32,[batch_sizes[1],image_dims[0],image_dims[1],image_dims[2]], name="anchor_val")
    positive_val = tf.placeholder(tf.float32, [batch_sizes[1],image_dims[0],image_dims[1],image_dims[2]], name="positive_val")
    negative_val = tf.placeholder(tf.float32,[batch_sizes[0],image_dims[0],image_dims[1],image_dims[2]], name="negative_val")
#    label_val = tf.placeholder(tf.float32, [batch_sizes[1],1], name="label_val") # 1 if same, 0 if different
    
    left_test = tf.placeholder(tf.float32, [batch_sizes[2],image_dims[0],image_dims[1],image_dims[2]], name="left_test")
    right_test = tf.placeholder(tf.float32, [batch_sizes[2],image_dims[0],image_dims[1],image_dims[2]], name="right_test")
    
#    return anchor_train,positive_train,label_train,anchor_val,positive_val,label_val,anchor_test,positive_test
    return anchor_train,positive_train,negative_train,anchor_val,positive_val,negative_val,left_test,right_test
    