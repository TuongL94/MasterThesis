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

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm/(1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
     
def kernel_tile(input, kernel_size, strides, padding):
    input_shape = input.get_shape()
    size = input_shape[3]*input_shape[4]
    input = tf.reshape(input, shape = [-1, input_shape[1], input_shape[2], size])
    input_shape = input.get_shape()
    tile_filter = np.zeros(shape=[kernel_size, kernel_size, input_shape[3], kernel_size*kernel_size], dtype=np.float32)
    
    # create convolutional filters
    for i in range(kernel_size):
        for j in range(kernel_size):
            tile_filter[i, j, :, i*kernel_size + j] = 1.0
    
    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[1, strides[0], strides[1], 1], padding=padding)
    output_shape = output.get_shape()
    output = tf.reshape(output, shape=[-1,output_shape[1], output_shape[2], input_shape[3], kernel_size* kernel_size])
    output = tf.transpose(output, perm=[0,1,2,4,3])
    return output

def mat_transform(input, output_cap_size, output_cap_dim, spatial_size, batch_size):
    input_shape = input.get_shape().as_list()

    def W_shared():
        with tf.variable_scope("W_shared", reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W",
                            trainable=True, 
                            shape=[1, input_shape[1], output_cap_size*spatial_size*spatial_size, output_cap_dim, input_shape[-1]], 
                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32))
        return W
    W = W_shared() 
    W_tiled = tf.tile(W, [batch_size,1,1,1,1], name="W_tiled")
    
    # add additional dimensions to output of caps1 to conform with the dimensions of W_tiled
    caps_output_expanded = tf.expand_dims(input, axis=-1, name="output_expanded")
    caps_output_tile = tf.expand_dims(caps_output_expanded, axis=2, name="output_tile")
    caps_output_tiled = tf.tile(caps_output_tile, [1,1,output_cap_size*spatial_size*spatial_size,1,1], name="output_tiled")
    
    caps_predicted = tf.matmul(W_tiled, caps_output_tiled, name="caps_predictions")
    return caps_predicted

def dynamic_routing(input, batch_size, routing_itr):
    input_shape = input.get_shape()
    
    def condition(b_coeff, counter):
        return tf.less(counter, 100)

    def loop_body(b_coeff, counter):
        routing_weights = tf.nn.softmax(b_coeff, axis=2, name="routing_weights")
        weighted_predictions = tf.multiply(routing_weights, input, name="weighted_predictions")
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True, name="weighted_sum")
        v = squash(weighted_sum, axis=-2)
        v_tiled = tf.tile(v, [1, input_shape[1], 1, 1, 1], name="v_tiled")
        agreement = tf.matmul(input, v_tiled, transpose_a=True, name="agreement")
        new_b_coeff = tf.add(b_coeff, agreement)
        return [new_b_coeff,tf.add(1,counter)]
    
    counter = 1
    b_0 = tf.zeros([batch_size, input_shape[1], input_shape[2], 1, 1],dtype=tf.float32)
    b_final = tf.while_loop(condition, loop_body, [b_0, counter], maximum_iterations=routing_itr)
    routing_weights = tf.nn.softmax(b_final[0], axis=2)
    weighted_predictions = tf.multiply(routing_weights, input)
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True)
    output = squash(weighted_sum, axis=-2)
    return output
     
def primary_caps(input, kernel_size, capsules, cap_dim, strides, padding="valid", name="primary_caps"):
    with tf.variable_scope(name):
        net = tf.layers.conv2d(
            inputs = input,
            filters = capsules*cap_dim,
            kernel_size = kernel_size,
            strides = strides,
            padding = padding,
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            name = "conv")
        
    net_shape = net.get_shape().as_list()    
    net = tf.reshape(net,[-1,net_shape[1],net_shape[2],capsules,cap_dim])
    return net
    
def conv_capsule(input, kernel_size, capsules, cap_dim, strides, padding, batch_size, routing_itr=3, name="conv_capsule"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        input_shape = input.get_shape().as_list()
        u_conv = kernel_tile(input, kernel_size, strides, padding)
        spatial_size = u_conv.get_shape().as_list()[1]
        
        u_conv = tf.reshape(u_conv, shape=[-1, kernel_size*kernel_size*spatial_size*spatial_size*input_shape[-2], input_shape[-1]])
        u_hat = mat_transform(u_conv, capsules, cap_dim, spatial_size, batch_size)
        net = dynamic_routing(u_hat, batch_size, routing_itr)
        net = tf.squeeze(net)
        net= tf.expand_dims(net, axis=1)
        net = tf.expand_dims(net, axis=1)
        net = tf.reshape(net, shape=[-1,spatial_size, spatial_size, capsules, cap_dim])
        return net
    
def capsule_net(input, output_size, output_dim, batch_size, name="capsule_net"):
    with tf.variable_scope(name):
        net = tf.layers.conv2d(
            inputs = input,
            filters = 64,
            kernel_size = [9,9], 
            strides = [2,2],
            padding = "valid",
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
            name="conv1")
        
        net = primary_caps(
                net,
                kernel_size=[9,9],
                capsules=8,
                cap_dim=4,
                strides=[2,2])
        
        net = conv_capsule(
                net,
                kernel_size = 3,
                capsules = output_size,
                cap_dim = output_dim,
                strides = [4,4],
                padding = "VALID",
                batch_size = batch_size,
                routing_itr = 2)
    
    return net
    
    