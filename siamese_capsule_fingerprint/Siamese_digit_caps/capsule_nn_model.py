#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:50:09 2018

@author: Tuong Lam & Simon Nilsson
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


import capsule_utility as cu


def primary_caps(input, kernel_size, capsules, cap_dim, strides, padding, name="primary_caps"):
    with tf.variable_scope(name):
        net = tf.layers.conv2d(
            inputs = input,
            filters = capsules*cap_dim,
            kernel_size = kernel_size,
            strides = strides,
            padding = padding,
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name = name)
        
#    net_shape = net.get_shape()
#    net = tf.reshape(net,[-1,net_shape[1]*net_shape[2]*capsules,cap_dim])
    return net
    
#def conv_capsule():
    
    
    
    
    
def capsule_net(input, routing_iterations, digit_caps_classes, digit_caps_dims, caps1_n_maps, caps1_n_dims, batch_size, name="capsule_net"):
    net = tf.layers.conv2d(
        inputs = input,
        filters = 32,
        kernel_size = [9,9], 
        strides = [2,2],
        padding = "valid",
        activation = tf.nn.relu,
        reuse = tf.AUTO_REUSE,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.1),
        name="conv1")
    
    net = tf.layers.batch_normalization(
        net,
#        training = training,
        name = "batch_norm_1",
        reuse = tf.AUTO_REUSE)
    
#    net = tf.layers.max_pooling2d(
#        inputs = net, 
#        pool_size = [2,2],
#        strides = 2)
    
    net = tf.layers.conv2d(
        inputs = net,
        filters = 32,
        kernel_size = [9,9], 
        strides = [2,2],
        padding = "valid",
        activation = tf.nn.relu,
        reuse = tf.AUTO_REUSE,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.1),
        name="conv2")
    
    net = tf.layers.batch_normalization(
        net,
#        training = training,
        name = "batch_norm_2",
        reuse = tf.AUTO_REUSE)
    
    net = primary_caps(
            net,
            kernel_size = [9,9],
            capsules = caps1_n_maps,
            cap_dim = caps1_n_dims,
            strides = [2,2],
            padding = "valid")
    
#    net = tf.layers.batch_normalization(
#        net,
##        training = training,
#        name = "batch_norm_3",
#        reuse = tf.AUTO_REUSE)
    
    
    ######### Starting routing procedure ##########################
    net_shape = net.get_shape().as_list()
    caps1_n_caps = caps1_n_maps * net_shape[1] * net_shape[2]
    
    caps1_raw = tf.reshape(net,[-1,caps1_n_caps,caps1_n_dims],name="caps1_raw") ####### OBS! Need to change network to output comaptible number of pararmeters to do reshape to capsules
    caps1_output = cu.squash(caps1_raw, name="caps1_output")
    
    def W_shared():
        with tf.variable_scope("W_shared", reuse=tf.AUTO_REUSE):
#            W_init = tf.random_normal(shape=[1, caps1_n_caps, digit_caps_classes, digit_caps_dims, caps1_n_dims],
#                              stddev=0.1, dtype=tf.float32, name = "W_init")
#            W = tf.get_variable(W_init,trainable=True, name="W") # transformation matrices, will be trained with backpropagation
            W = tf.get_variable("W",
                            trainable=True, 
                            shape=[1, caps1_n_caps, digit_caps_classes, digit_caps_dims, caps1_n_dims], 
                            initializer=tf.random_normal_initializer(mean=0.0, stddev=1.5, dtype=tf.float32))
        return W
    W = W_shared()    
    W_tiled = tf.tile(W, [batch_size,1,1,1,1], name="W_tiled")
    
    # add additional dimensions to output of caps1 to conform with the dimensions of W_tiled
    caps1_output_expanded = tf.expand_dims(caps1_output, axis=-1, name="caps1_output_expanded")
    caps1_output_tile = tf.expand_dims(caps1_output_expanded, axis=2, name="caps1_output_tile")
    caps1_output_tiled = tf.tile(caps1_output_tile, [1,1,digit_caps_classes,1,1], name="caps1_output_tiled")
    
    caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")
    
    nbr_of_routing_iterations = 2
    counter = 1
    b_0 = tf.zeros([batch_size, caps1_n_caps, digit_caps_classes, 1, 1],dtype=tf.float32)

    def condition(b_coeff, counter):
        return tf.less(counter, 100)

    def loop_body(b_coeff, counter):
        routing_weights = tf.nn.softmax(b_coeff, axis=2, name="routing_weights")
        weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True, name="weighted_sum")
        v = cu.squash(weighted_sum, axis=-2)
        v_tiled = tf.tile(v, [1, caps1_n_caps, 1, 1, 1], name="v_tiled")
        agreement = tf.matmul(caps2_predicted, v_tiled, transpose_a=True, name="agreement")
        new_b_coeff = tf.add(b_coeff, agreement)
        return [new_b_coeff,tf.add(1,counter)]
    
    b_final = tf.while_loop(condition, loop_body, [b_0, counter], maximum_iterations=nbr_of_routing_iterations)
        
    routing_weights = tf.nn.softmax(b_final[0], axis=2)
    weighted_predictions = tf.multiply(routing_weights, caps2_predicted)
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True)
    caps2_output = cu.squash(weighted_sum, axis=-2)
    
    # Create reconstruction network
#    image_size = input.get_shape()
#    reconstruct = reconstruction_net(caps2_output, image_size[1:3])

#    total_parameters = 0
#    for variable in tf.trainable_variables():
#        # shape is an array of tf.Dimension
#        shape = variable.get_shape()
#        print(shape)
#        print(len(shape))
#        variable_parameters = 1
#        for dim in shape:
#            print(dim)
#            variable_parameters *= dim.value
#        print(variable_parameters)
#        total_parameters += variable_parameters
#    print(total_parameters)

    return caps2_output



def reconstruction_net(digit_caps, image_size, name="Reconstruction_net"):
    caps_flat = tf.layers.flatten(digit_caps)
    
    net = tf.layers.dense(
            caps_flat,
            512,
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            name = "recon1")
    
    net = tf.layers.dense(
            net,
            1024,
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            name = "recon2")

    net = tf.layers.dense(
            net,
            image_size[0] * image_size[1],
            activation = tf.nn.sigmoid,
            reuse = tf.AUTO_REUSE,
            name = "reconstruction_output")

    return net









    
    