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

def inception_a_block(input):
    conv1 = tf.layers.conv2d(
        inputs = input,
        filters = 16,
        kernel_size = [1,1], 
        padding = "same",
        activation = tf.nn.relu,
        reuse = tf.AUTO_REUSE,
        kernel_initializer = tf.initializers.truncated_normal(mean=0, stddev=0.2),
        name="conv1_layer_1")
    
    conv2 = tf.layers.conv2d(
        inputs = input,
        filters = 16,
        kernel_size = [3,3], 
        padding = "same",
        activation = tf.nn.relu,
        reuse = tf.AUTO_REUSE,
        kernel_initializer = tf.initializers.truncated_normal(mean=0, stddev=0.2),
        name="conv2_layer_1")
    
    conv3 = tf.layers.conv2d(
        inputs = input,
        filters = 16,
        kernel_size = [5,5], 
        padding = "same",
        activation = tf.nn.relu,
        reuse = tf.AUTO_REUSE,
        kernel_initializer = tf.initializers.truncated_normal(mean=0, stddev=0.2),
        name="conv3_layer_1")
    
    output = tf.concat([conv1,conv2,conv3],axis=3)
    return output

def inference(input):
    output = inception_a_block(input)
    output = tf.layers.flatten(output)
    output = tf.layers.dense(
            output,
            10,
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="dense_1")
    return output