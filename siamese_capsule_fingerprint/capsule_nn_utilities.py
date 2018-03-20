#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:02:14 2018

@author: Tuong Lam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False, name=None):
    with tf.name_scope(name, default_name="safe_2_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
        return tf.sqrt(squared_norm)
    
def scaled_pair_loss(input_1, input_2, label, epsilon=1e-7):
    norm_1 = safe_norm(input_1, axis=-1)
    norm_2 = safe_norm(input_2, axis=-1)
    diff = safe_norm(input_1 - input_2)
    loss_match = tf.reduce_sum(tf.truediv(1.0,norm_1 + norm_2 + epsilon) * diff)
    loss_no_match = tf.reduce_sum((norm_1 + norm_2) * diff)
    loss = tf.reduce_sum(label * loss_match + (1-label) * loss_no_match)
    return loss

def momentum_training(loss, learning_rate, momentum):
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum, use_nesterov=True)
    train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op

def adadelta_training(loss,learning_rate,rho,epsilon):
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,rho=rho,epsilon=epsilon)
    train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op
    
def placeholder_inputs(image_dims,batch_sizes):
    """ Creates placeholders for capsule neural network.
    
    This method returns placeholders for inputs to the capsule network in both
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
