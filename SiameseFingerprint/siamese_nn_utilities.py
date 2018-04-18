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

def contrastive_loss(input_1,input_2,label,margin):
    """ Computes the contrastive loss between two vectors
    
    Input:
    input_1 - first input vector
    input_2 - second input vector
    label - ground truth for similarity between the vectors. 1 if they are similar, 0 otherwise.
    margin - margin for contrastive loss, positive constant
    Returns the contrastive loss between input_1 and input_2 with specified margin.
    """
    d_sq = tf.reduce_sum(tf.pow(input_1-input_2,2),1,keepdims=True)
    max_sq = tf.square(tf.maximum(margin-d_sq,0))
    return tf.reduce_mean(label*d_sq + (1-label)*max_sq)/2

def cross_entropy_loss(input, label, pos_weight):
    losses = tf.nn.weighted_cross_entropy_with_logits(label, input, pos_weight)
    return tf.reduce_mean(losses, keepdims=False)
    
def momentum_training(loss, learning_rate, momentum):
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op

def adadelta_training(loss,learning_rate,rho,epsilon):
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,rho=rho,epsilon=epsilon)
    train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op
    
def create_placeholders(image_dims):
    """ Creates placeholders for siamese neural network.
    
    This method returns placeholders for inputs to the siamese network in both
    training, validation and testing. It also returns a placeholder for ground truth of 
    image pairs for training and validation.
    Input:
    image_dims - list of following structure: [height width 1] (the 1 is for grayscale images)
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
    left_train = tf.placeholder(tf.float32, [None, image_dims[0], image_dims[1], image_dims[2]], name="left_train")
    right_train = tf.placeholder(tf.float32, [None, image_dims[0], image_dims[1], image_dims[2]], name="right_train")
    label_train = tf.placeholder(tf.float32, [None, 1], name="label_train") # 1 if same, 0 if different
    
    left_val = tf.placeholder(tf.float32, [None, image_dims[0], image_dims[1], image_dims[2]], name="left_val")
    right_val = tf.placeholder(tf.float32, [None, image_dims[0], image_dims[1], image_dims[2]], name="right_val")
    label_val = tf.placeholder(tf.float32, [None, 1], name="label_val") # 1 if same, 0 if different
    
    left_test = tf.placeholder(tf.float32, [None, image_dims[0], image_dims[1], image_dims[2]], name="left_test")
    right_test = tf.placeholder(tf.float32, [None, image_dims[0], image_dims[1], image_dims[2]], name="right_test")
    return left_train,right_train,label_train,left_val,right_val,label_val,left_test,right_test