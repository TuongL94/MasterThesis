#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:50:09 2018

@author: Tuong Lam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

def primary_caps(input, kernel_size, capsules, strides, padding, name):
    net = tf.layers.conv2d(
        inputs = input,
        filters = caps1_n_maps*caps1_n_dims,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = tf.nn.relu,
        name = name)
    
    net = tf.reshape(net,[-1,caps1_n_caps,caps1_n_dims])
    
    
        
def capsule_net(input, output_size, routing_iterations, name="capsule_net"):
    net = tf.layers.conv2d(
        inputs = input,
        filters = 256,
        kernel_size = [9,9], 
        strides = [2,2],
        padding = "valid",
        activation = tf.nn.relu,
        reuse = tf.AUTO_REUSE,
        name="conv1")
    
    net = primary_caps(
            net,
            kernel_size=[1,1],
            capsules = 32,
            strides = [2,2],
            padding = "valid")
    
    net = conv_capsule(
            net,
            kernel_size,
            capsules,
            strides,
            padding)
    
    return net
    
    