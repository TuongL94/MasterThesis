#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:57:15 2018

@author: Tuong Lam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def inception_a_block(input, training):
    conv1 = tf.layers.conv2d(
        inputs = input,
        filters = 16,
        kernel_size = [1,1], 
        padding = "same",
        activation = tf.nn.leaky_relu,
        reuse = tf.AUTO_REUSE,
        kernel_initializer = tf.initializers.truncated_normal(mean=0, stddev=0.2),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
        name="conv1")
    
    conv2 = tf.layers.conv2d(
        inputs = input,
        filters = 16,
        kernel_size = [3,3], 
        padding = "same",
        activation = tf.nn.leaky_relu,
        reuse = tf.AUTO_REUSE,
        kernel_initializer = tf.initializers.truncated_normal(mean=0, stddev=0.2),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
        name="conv2")
    
    conv3 = tf.layers.conv2d(
        inputs = input,
        filters = 16,
        kernel_size = [5,5], 
        padding = "same",
        activation = tf.nn.leaky_relu,
        reuse = tf.AUTO_REUSE,
        kernel_initializer = tf.initializers.truncated_normal(mean=0, stddev=0.2),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
        name="conv3")
    
    output = tf.concat([conv1,conv2,conv3],axis=3)
    return output

def stem(input, training):
    
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
    
    output = tf.layers.dropout(
            output,
            rate = 0.5,
            training = training)
        
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
    
    output = tf.layers.dropout(
            output,
            rate = 0.5,
            training = training)
        
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
    
    output = tf.layers.dropout(
            output,
            rate = 0.5,
            training = training)
              
    return output

def inference(input, training = True):
    output = stem(input, training)
    with tf.variable_scope("inception_1"):
        output = inception_a_block(input, training)
        
    output = tf.layers.max_pooling2d(inputs = output, 
                                     pool_size = [2,2], 
                                     strides = 2)
    
    with tf.variable_scope("inception_2"):
        output = inception_a_block(output, training)
                
    output = tf.layers.flatten(output)
    output = tf.layers.dense(
            output,
            10,
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="dense")
    output = tf.layers.dropout(
            output,
            rate = 0.2,
            training = training)
    output = tf.nn.l2_normalize(
            output,
            axis=1)
    return output