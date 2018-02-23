# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:55:36 2018

@author: Tuong Lam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def conv_relu_fixed(input,kernel,bias,trainable = False):
    # Create variable named "weight".
    weight = tf.get_variable("weight", initializer = kernel,
        trainable = trainable)
    # Create variable named "bias".
    bias = tf.get_variable("bias", initializer = bias,
        trainable = trainable)
    conv = tf.nn.conv2d(input, weight,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + bias)

def conv_relu(input,kernel_shape,bias_shape):
    # Create variable named "weight".
    weight = tf.get_variable("weight",shape=kernel_shape,
        trainable = True)
    # Create variable named "bias".
    bias = tf.get_variable("bias",shape=bias_shape,
        trainable = True)
    conv = tf.nn.conv2d(input, weight,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + bias)

def inference(input,*transfer):
    input_layer = input
    
    # Convolutional layer 1
    with tf.variable_scope("conv1"):
        if len(transfer) == 1:
            conv1 = conv_relu_fixed(
                    input_layer,
                    transfer[0][0],
                    transfer[0][1])
        else:
            conv1 = conv_relu(
                    input_layer,
                    [5,5,1,32],
                    [32])

    # Pooling layer 1
    pool1 = tf.nn.max_pool(
            value = conv1,
            ksize = [1,2,2,1], 
            strides = [1,2,2,1],
            padding = "SAME")
    
    # Convolutional layer 2
    with tf.variable_scope("conv2"):
        conv2 = conv_relu(
                pool1,
                [5,5,32,64],
                [64])
            
    # Pooling layer 2
    pool2 = tf.nn.max_pool(
            value = conv2,
            ksize = [1,2,2,1], 
            strides = [1,2,2,1],
            padding = "SAME")

    net = tf.layers.flatten(pool2)
    
    # Fully connected layer 1
    net = tf.layers.dense(
            inputs = net,
            units = 1024,
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            name = "dense_layer_1")
    
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
        
def training(loss, learning_rate, momentum):
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum, use_nesterov=True)
    train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op
    
def placeholder_inputs(image_dims,batch_sizes):
    """ Creates placeholders for siamese neural network.
    
    This method returns placeholders for inputs to the siamese network in both
    training, validation and testing. It also returns a placeholder for ground truth of 
    image pairs for training and validation.
    Input:
    image_dims - list of following structure: [height width 1] (the 1 is for grayscale images)
    batch_sizes - list of batch sizes for training,validation and testing respectively 
    Returns:
    left_train - placeholder for left input to siamese network for training
    right_train - placeholder for right input to siamese network for training
    label_train - placeholder for ground truths of image pairs for training
    left_val - placeholder for left input to siamese network for validation
    right_val - placeholder for right input to siamese network for validation
    label_val - placeholder for ground truths of image pairs for validation
    left_test - placeholder for left input to siamese network for testing
    right_test - placeholder for right input to siamese network for testing
    """
    left_train = tf.placeholder(tf.float32, [batch_sizes[0],image_dims[0],image_dims[1],image_dims[2]], name="left_train")
    right_train = tf.placeholder(tf.float32,[batch_sizes[0],image_dims[0],image_dims[1],image_dims[2]], name="right_train")
    label_train = tf.placeholder(tf.float32, [batch_sizes[0],1], name="label_train") # 1 if same, 0 if different
    
    left_val = tf.placeholder(tf.float32,[batch_sizes[1],image_dims[0],image_dims[1],image_dims[2]], name="left_val")
    right_val = tf.placeholder(tf.float32, [batch_sizes[1],image_dims[0],image_dims[1],image_dims[2]], name="right_val")
    label_val = tf.placeholder(tf.float32, [batch_sizes[1],1], name="label_val") # 1 if same, 0 if different
    
    left_test = tf.placeholder(tf.float32, [batch_sizes[2],image_dims[0],image_dims[1],image_dims[2]], name="left_test")
    right_test = tf.placeholder(tf.float32, [batch_sizes[2],image_dims[0],image_dims[1],image_dims[2]], name="right_test")
    return left_train,right_train,label_train,left_val,right_val,label_val,left_test,right_test
    