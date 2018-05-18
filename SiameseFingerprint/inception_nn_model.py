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
    
    conv1_1 = tf.layers.conv2d(
        inputs = input,
        filters = 32,
        kernel_size = [1,1], 
        padding = "same",
        reuse = tf.AUTO_REUSE,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
        name="conv1_1")
    
    conv1_1 = tf.layers.batch_normalization(
            conv1_1,
            training = training,
            name = "batch_norm_1_1",
            reuse = tf.AUTO_REUSE)
    
    conv1_1 = tf.nn.leaky_relu(
            conv1_1)
    
    conv2_1 = tf.layers.conv2d(
        inputs = input,
        filters = 32,
        kernel_size = [1,1], 
        padding = "same",
        reuse = tf.AUTO_REUSE,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
        name="conv2_1")
    
    conv2_1 = tf.layers.batch_normalization(
            conv2_1,
            training = training,
            name = "batch_norm_2_1",
            reuse = tf.AUTO_REUSE)
    
    conv2_1 = tf.nn.leaky_relu(
            conv2_1)
    
    conv3_1 = tf.layers.conv2d(
        inputs = input,
        filters = 32,
        kernel_size = [1,1], 
        padding = "same",
        reuse = tf.AUTO_REUSE,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
        name="conv3_1")
    
    conv3_1 = tf.layers.batch_normalization(
            conv3_1,
            training = training,
            name = "batch_norm_3_1",
            reuse = tf.AUTO_REUSE)
    
    conv3_1 = tf.nn.leaky_relu(
            conv3_1)
    
    max1 = tf.layers.max_pooling2d(inputs = input, 
                                   pool_size = [3,3], 
                                   strides = 1,
                                   padding = "same")  
    conv4_1 = tf.layers.conv2d(
        inputs = max1,
        filters = 32,
        kernel_size = [1,1], 
        padding = "same",
        reuse = tf.AUTO_REUSE,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
        name="conv4_1")
    
    conv4_1 = tf.layers.batch_normalization(
            conv4_1,
            training = training,
            name = "batch_norm_4_1",
            reuse = tf.AUTO_REUSE)
    
    conv4_1 = tf.nn.leaky_relu(
            conv4_1)
   
    conv2_2 = tf.layers.conv2d(
        inputs = conv2_1,
        filters = 32,
        kernel_size = [3,3], 
        padding = "same",
        reuse = tf.AUTO_REUSE,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
        name="conv2_2")
    
    conv2_2 = tf.layers.batch_normalization(
            conv2_2,
            training = training,
            name = "batch_norm_2_2",
            reuse = tf.AUTO_REUSE)
        
    conv2_2 = tf.nn.leaky_relu(
            conv2_2)
    
    conv3_2 = tf.layers.conv2d(
        inputs = conv3_1,
        filters = 32,
        kernel_size = [3,3], 
        padding = "same",
        reuse = tf.AUTO_REUSE,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
        name="conv3_2")
    
    conv3_2 = tf.layers.batch_normalization(
            conv3_2,
            training = training,
            name = "batch_norm_3_2",
            reuse = tf.AUTO_REUSE)
        
    conv3_2 = tf.nn.leaky_relu(
            conv3_2)
    
    conv3_3 = tf.layers.conv2d(
        inputs = conv3_2,
        filters = 32,
        kernel_size = [3,3], 
        padding = "same",
        reuse = tf.AUTO_REUSE,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
        name="conv3_3")
    
    conv3_3 = tf.layers.batch_normalization(
            conv3_3,
            training = training,
            name = "batch_norm_3_3",
            reuse = tf.AUTO_REUSE)
        
    conv3_3 = tf.nn.leaky_relu(
            conv3_3)
    
    output = tf.concat([conv1_1,conv4_1,conv2_2,conv3_3],axis=3)
    return output

def inception_b_block(input, training):
    
    conv1_1 = tf.layers.conv2d(
            inputs = input,
            filters = 32,
            kernel_size = [1,1],
            strides = [1,1],
            padding = "same",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv1_1")
    
    conv1_1 = tf.layers.batch_normalization(
            conv1_1,
            training = training,
            name = "batch_norm_1_1",
            reuse = tf.AUTO_REUSE)
    
    conv1_1 = tf.nn.leaky_relu(
            conv1_1)
    
    conv2_1 = tf.layers.conv2d(
            inputs = input,
            filters = 32,
            kernel_size = [1,1],
            strides = [1,1],
            padding = "same",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv2_1")
    
    conv2_1 = tf.layers.batch_normalization(
            conv2_1,
            training = training,
            name = "batch_norm_2_1",
            reuse = tf.AUTO_REUSE)
    
    conv2_1 = tf.nn.leaky_relu(
            conv2_1)
    
    conv3_1 = tf.layers.conv2d(
            inputs = input,
            filters = 32,
            kernel_size = [1,1],
            strides = [1,1],
            padding = "same",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv3_1")
    
    conv3_1 = tf.layers.batch_normalization(
            conv3_1,
            training = training,
            name = "batch_norm_3_1",
            reuse = tf.AUTO_REUSE)
    
    conv3_1 = tf.nn.leaky_relu(
            conv3_1)
    
    conv1_2 = tf.layers.conv2d(
            inputs = conv1_1,
            filters = 64,
            kernel_size = [3,3],
            strides = [1,1],
            padding = "same",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv1_2")
    
    conv1_2 = tf.layers.batch_normalization(
            conv1_2,
            training = training,
            name = "batch_norm_1_2",
            reuse = tf.AUTO_REUSE)
    
    conv1_2 = tf.nn.leaky_relu(
            conv1_2)
    
    conv2_2 = tf.layers.conv2d(
            inputs = conv2_1,
            filters = 64,
            kernel_size = [3,3],
            strides = [1,1],
            padding = "same",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv2_2")
    
    conv2_2 = tf.layers.batch_normalization(
            conv2_2,
            training = training,
            name = "batch_norm_2_2",
            reuse = tf.AUTO_REUSE)
    
    conv2_2 = tf.nn.leaky_relu(
            conv2_2)
    
    conv3_2 = tf.layers.conv2d(
            inputs = conv3_1,
            filters = 64,
            kernel_size = [3,3],
            strides = [1,1],
            padding = "same",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv3_2")
    
    conv3_2 = tf.layers.batch_normalization(
            conv3_2,
            training = training,
            name = "batch_norm_3_2",
            reuse = tf.AUTO_REUSE)
    
    conv3_2 = tf.nn.leaky_relu(
            conv3_2)
    
    output = tf.concat([conv1_2,conv2_2,conv3_2],axis=3)
    return output
    
    
def reduction_1_block(input, training):
    
    max1_1 = tf.layers.max_pooling2d(inputs = input, 
                                     pool_size = [3,3], 
                                     strides = 2)
    conv1_1 = tf.layers.conv2d(
            inputs = input,
            filters = 64,
            kernel_size = [3,3],
            strides = [2,2],
            padding = "valid",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv1_1")
    
    conv1_1 = tf.layers.batch_normalization(
            conv1_1,
            training = training,
            name = "batch_norm_1_1",
            reuse = tf.AUTO_REUSE)
    
    conv1_1 = tf.nn.leaky_relu(
            conv1_1)
    
    conv2_1 =  tf.layers.conv2d(
            inputs = input,
            filters = 64,
            kernel_size = [1,1],
            strides = [1,1],
            padding = "same",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv2_1")
    
    conv2_1 = tf.layers.batch_normalization(
            conv2_1,
            training = training,
            name = "batch_norm_2_1",
            reuse = tf.AUTO_REUSE)
    
    conv2_1 = tf.nn.leaky_relu(
            conv2_1)
    
    conv2_2 =  tf.layers.conv2d(
            inputs = conv2_1,
            filters = 64,
            kernel_size = [3,3],
            strides = [1,1],
            padding = "same",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv2_2")
    
    conv2_2 = tf.layers.batch_normalization(
            conv2_2,
            training = training,
            name = "batch_norm_2_2",
            reuse = tf.AUTO_REUSE)
    
    conv2_2 = tf.nn.leaky_relu(
            conv2_2)
    
    conv3_2 =  tf.layers.conv2d(
            inputs = conv2_2,
            filters = 64,
            kernel_size = [3,3],
            strides = [2,2],
            padding = "valid",
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="conv3_2")
    
    conv3_2 = tf.layers.batch_normalization(
            conv3_2,
            training = training,
            name = "batch_norm_3_2",
            reuse = tf.AUTO_REUSE)
    
    conv3_2 = tf.nn.leaky_relu(
            conv3_2)
    
    output = tf.concat([conv1_1,max1_1,conv3_2],axis=3)
    
    return output

def stem(input, training):
    
    output = tf.layers.batch_normalization(
            input,
            training = training,
            name = "batch_norm_1",
            reuse = tf.AUTO_REUSE)
    
    # Convolutional layer 1
    output = tf.layers.conv2d(
            inputs = output,
            filters = 16,
            kernel_size = [7,7],
            strides = [1,1],
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
            filters = 16,
            kernel_size = [5,5],
            strides = [1,1],
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
    
#    # Convolutional Layer 3
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 32,
#            kernel_size = [3,3],
#            strides = [1,1],
#            padding = "same",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv3")
#            
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_4",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 3)
    
#     # Convolutional Layer 4
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 64,
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
#        seed = 4)
                      
    return output

def inference(input, training = True):
    output = stem(input, training)
    
    with tf.variable_scope("inception_1"):
        output = inception_a_block(output, training)
    
    with tf.variable_scope("inception_2"):
        output = inception_a_block(output, training)
    
    with tf.variable_scope("inception_3"):
        output = inception_a_block(output, training)
        
    with tf.variable_scope("reduction_1"):
        output = reduction_1_block(output, training)
            
    with tf.variable_scope("inception_4"):
        output = inception_b_block(output, training)  
        
    output = tf.layers.max_pooling2d(inputs = output, 
                                     pool_size = [2,2], 
                                     strides = 2)
              
    output = tf.layers.flatten(output)
    
    output = tf.layers.dense(
            output,
            512,
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
            name="dense")
    
    output = tf.nn.leaky_relu(
            output)

    output = tf.nn.l2_normalize(
            output,
            axis=1)
    
    return output