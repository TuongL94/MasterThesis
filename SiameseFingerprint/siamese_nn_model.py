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
            filters = 2,
            kernel_size = [3,3], 
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=0.1),
#            kernel_initializer = tf.random_uniform_initializer(minval=-0, maxval=0),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0.5, stddev=0.25),
            name="conv_layer_1") 
        
    # Pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, 
                                     pool_size = [2,2], 
                                     strides = 2)
    # Convolutional layer 1
    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 2,
            kernel_size = [3,3], 
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#                kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=4.0),
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0.3, stddev=0.1),
            name="conv_layer_2")
    
    # Convolutional layer 1
    conv3 = tf.layers.conv2d(
            inputs = conv2,
            filters = 2,
            kernel_size = [3,3], 
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0.3, stddev=0.1),
            name="conv_layer_3")
    
    # Convolutional layer 1
    conv4 = tf.layers.conv2d(
            inputs = conv3,
            filters = 2,
            kernel_size = [3,3], 
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0.2, stddev=0.1),
            name="conv_layer_4")
    
    # Convolutional layer 2
    conv5 = tf.layers.conv2d(
            inputs = conv4,
            filters = 2,
            kernel_size = [3,3],
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0.2, stddev=0.1),
            name="conv_layer_5")
        
    # Pooling layer 2   
    pool2 = tf.layers.max_pooling2d(
            inputs = conv5, 
            pool_size = [2,2],
            strides = 2)
    
    # Convolutional layer 3
    conv6 = tf.layers.conv2d(
            inputs = pool2,
            filters = 4,
            kernel_size = [3,3],
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0.2, stddev=0.1),
            name="conv_layer_6")
            
    # Pooling layer 3
    pool3 = tf.layers.max_pooling2d(
        inputs = conv6, 
        pool_size = [2,2],
        strides = 2)
    
    # Convolutional Layer 4
    conv7 = tf.layers.conv2d(
            inputs = pool3,
            filters = 4,
            kernel_size = [3,3],
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0.2, stddev=0.1),
            name="conv_layer_7")
            
    pool4 = tf.layers.max_pooling2d(
            inputs = conv7, 
            pool_size = [2,2],
            strides = 2)
    
     # Convolutional layer 1
    conv8 = tf.layers.conv2d(
            inputs = pool4,
            filters = 4,
            kernel_size = [3,3], 
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.05),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.1),
            name="conv_layer_8")
    
     # Convolutional layer 1
    conv9 = tf.layers.conv2d(
            inputs = conv8,
            filters = 4,
            kernel_size = [3,3], 
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.05),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.1),
            name="conv_layer_9")
    
     # Convolutional layer 1
    conv10 = tf.layers.conv2d(
            inputs =  conv9,
            filters = 4,
            kernel_size = [3,3], 
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.05),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0, stddev=0.1),
            name="conv_layer_10")
    
     # Convolutional layer 1
    conv11 = tf.layers.conv2d(
            inputs =  conv10,
            filters = 8,
            kernel_size = [3,3], 
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.05),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0, stddev=0.1),
            name="conv_layer_11")
    
     # Convolutional layer 1
    conv12 = tf.layers.conv2d(
            inputs =  conv11,
            filters = 8,
            kernel_size = [3,3], 
            padding = "same",
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(mean=0.1, stddev=0.05),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0, stddev=0.1),
            name="conv_layer_12")
    
    net = tf.layers.flatten(conv12)
    
    net = tf.layers.dense(
            net,
            512,
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            bias_initializer = tf.initializers.truncated_normal(mean=0.5, stddev=0.1),
            name="dense_layer_1")
    
    net = tf.layers.dropout(
            inputs = net,
            rate = 0.7)
     
    return net
    