# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:55:36 2018

@author: Tuong Lam & Simon Nilsson
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

#def inference(input, training = True):
#    
#    
#    output = tf.layers.conv2d(
#            inputs = input,
#            filters = 12,
#            kernel_size = [5,5], 
#            strides = [1,1],
#            padding = "valid",
#            activation = tf.nn.relu,
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.5),
#            name="conv1")
#    
#    output = tf.layers.max_pooling2d(
#            inputs = output,
#            pool_size = [2,2],
#            strides = 2,
#            name = "pool1")
#    
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 12,
#            kernel_size = [5,5], 
#            strides = [1,1],
#            padding = "valid",
#            activation = tf.nn.relu,
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.5),
#            name="conv2")
#    
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 16,
#            kernel_size = [5,5], 
#            strides = [1,1],
#            padding = "valid",
#            activation = tf.nn.relu,
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.5),
#            name="conv3")
#    
#    output = tf.layers.max_pooling2d(
#            inputs = output,
#            pool_size = [2,2],
#            strides = 2,
#            name = "pool2")
#    
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 24,
#            kernel_size = [5,5], 
#            strides = [1,1],
#            padding = "valid",
#            activation = tf.nn.relu,
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.5),
#            name="conv4")
#    
##    # Convolutional layer 1
##    output = tf.layers.conv2d(
##            inputs = input,
##            filters = 16,
##            kernel_size = [7,7],
##            strides = [2,2],
##            padding = "same",
##            activation = tf.nn.leaky_relu,
##            reuse = tf.AUTO_REUSE,
###            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
##            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
##            name="conv1") 
##        
##    # Pooling layer 1
##    output = tf.layers.max_pooling2d(inputs = output, 
##                                     pool_size = [2,2], 
##                                     strides = 2)
##    
##    # Convolutional Layer 2
##    output = tf.layers.conv2d(
##            inputs = output,
##            filters = 16,
##            kernel_size = [5,5],
##            strides = [2,2],
##            padding = "same",
##            activation = tf.nn.leaky_relu,
##            reuse = tf.AUTO_REUSE,
###            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
##            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
##            name="conv2")
#    
##    output = tf.layers.dropout(
##            output,
##            rate = 0.5,
##            training = training)
#        
##    # Pooling layer 2
##    output = tf.layers.max_pooling2d(
##            inputs = output, 
##            pool_size = [2,2],
##            strides = 2)
##    
##    # Convolutional Layer 3
##    output = tf.layers.conv2d(
##            inputs = output,
##            filters = 32,
##            kernel_size = [3,3],
##            padding = "same",
##            activation = tf.nn.leaky_relu,
##            reuse = tf.AUTO_REUSE,
##            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
##            name="conv3")
##    
###    output = tf.layers.dropout(
###            output,
###            rate = 0.5,
###            training = training)
##        
##    # Convolutional Layer 4
##    output = tf.layers.conv2d(
##            inputs = output,
##            filters = 64,
##            kernel_size = [3,3],
##            padding = "same",
##            activation = tf.nn.leaky_relu,
##            reuse = tf.AUTO_REUSE,
##            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
##            name="conv4")
#    
#    output = tf.layers.flatten(
#            output)
#    
#    output = tf.layers.dense(
#        output,
#        1024,
#        activation = tf.nn.leaky_relu,
#        reuse = tf.AUTO_REUSE,
#        kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#        name="dense")
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training)
#    
#    output = tf.nn.l2_normalize(
#            output,
#            axis=1)
#              
#    return output



def inference(input, training = True):
    
    output = tf.layers.batch_normalization(
            input,
            training = training,
            name = "batch_norm_1",
            reuse = tf.AUTO_REUSE)
    
    # Convolutional layer 1
    output = tf.layers.conv2d(
            inputs = output,
            filters = 64,
            kernel_size = [7,7],
            strides = [2,2],
            padding = "valid",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv1") 
     
    output = tf.layers.batch_normalization(
        output,
        training = training,
        name = "batch_norm_2",
        reuse = tf.AUTO_REUSE)
    
    output = tf.nn.leaky_relu(
            output)
    
    output = tf.layers.dropout(
        output,
        rate = 0.5,
        training = training,
        seed = 1)
    
    # Pooling layer 1
    output = tf.layers.max_pooling2d(inputs = output, 
                                     pool_size = [2,2], 
                                     strides = 2)    
    # Convolutional Layer 2
    output = tf.layers.conv2d(
            inputs = output,
            filters = 64,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "valid",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv2")
        
    output = tf.layers.batch_normalization(
        output,
        training = training,
        name = "batch_norm_3",
        reuse = tf.AUTO_REUSE)
    
    output = tf.nn.leaky_relu(
            output)
    
    output = tf.layers.dropout(
        output,
        rate = 0.5,
        training = training,
        seed = 2)
    
    # Pooling layer 2
    output = tf.layers.max_pooling2d(
            inputs = output, 
            pool_size = [2,2],
            strides = 2)
    
    # Convolutional Layer 3
    output = tf.layers.conv2d(
            inputs = output,
            filters = 64,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "same",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv3")
            
    output = tf.layers.batch_normalization(
        output,
        training = training,
        name = "batch_norm_4",
        reuse = tf.AUTO_REUSE)
    
    output = tf.nn.leaky_relu(
            output)
    
    output = tf.layers.dropout(
        output,
        rate = 0.5,
        training = training,
        seed = 3)
    
    # Convolutional Layer 4
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 128,
#            kernel_size = [3,3],
#            strides = [1,1],
#            padding = "same",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv4")
#        
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_5",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed 111= 4)
    
    output = tf.layers.flatten(
            output)
    
    output = tf.layers.dense(
        output,
        1024,
        reuse = tf.AUTO_REUSE,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
        name="dense")
    
    output = tf.layers.batch_normalization(
        output,
        training = training,
        name = "batch_norm_6",
        reuse = tf.AUTO_REUSE)
    
    output = tf.nn.leaky_relu(
            output)
        
    output = tf.layers.dropout(
        output,
        rate = 0.5,
        training = training,
        seed = 5)
    
    output = tf.nn.l2_normalize(
            output,
            axis=1)
              
    return output
    