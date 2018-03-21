# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:55:36 2018

@author: Tuong Lam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def inference(input, training = True):
    
    # Convolutional layer 1
    output = tf.layers.conv2d(
            inputs = input,
            filters = 16,
            kernel_size = [7,7],
            strides = [2,2],
            padding = "valid",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv1") 
        
    # Pooling layer 1
    output = tf.layers.max_pooling2d(inputs = output, 
                                     pool_size = [2,2], 
                                     strides = 2)
    
    # Convolutional Layer 2
    output = tf.layers.conv2d(
            inputs = output,
            filters = 16,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "valid",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv2")
    
#    output = tf.layers.dropout(
#            output,
#            rate = 0.5,
#            training = training)
        
    # Pooling layer 2
    output = tf.layers.max_pooling2d(
            inputs = output, 
            pool_size = [2,2],
            strides = 2)
    
    # Convolutional Layer 3
    output = tf.layers.conv2d(
            inputs = output,
            filters = 32,
            kernel_size = [3,3],
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv3")
    
#    output = tf.layers.dropout(
#            output,
#            rate = 0.5,
#            training = training)
        
    # Convolutional Layer 4
    output = tf.layers.conv2d(
            inputs = output,
            filters = 64,
            kernel_size = [3,3],
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv4")
    
    output = tf.layers.flatten(
            output)
    
    output = tf.layers.dense(
        output,
        1024,
        activation = tf.nn.leaky_relu,
        reuse = tf.AUTO_REUSE,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
        name="dense")
        
    output = tf.nn.l2_normalize(
            output,
            axis=1)
              
    return output
    