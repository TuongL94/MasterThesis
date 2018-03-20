#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  19 09:02:14 2018

@author: Simon Nilsson
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

def margin_loss(caps_input, gt, m_plus=0.9, m_minus=0.1, lambda_=0.5):
    caps_input_norms = safe_norm(caps_input, axis=-2, keepdims=True)
    present_errors = tf.square(tf.maximum(0.0, m_plus - caps_input_norms))
    present_errors = tf.reshape(present_errors, shape=[-1,10])
    absent_errors = tf.square(tf.maximum(0.0, caps_input_norms - m_minus))
    absent_errors = tf.reshape(absent_errors, shape=[-1,10])
    loss = tf.add(gt * present_errors, lambda_ * (1.0 - gt) * absent_errors)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss

def scaled_pair_loss(input_1, input_2, label, epsilon=1e-7):
    norm_1 = safe_norm(input_1, axis=-2)
    norm_2 = safe_norm(input_2, axis=-2)
    diff = safe_norm(input_1 - input_2)
    loss_match = tf.truediv(1.0,norm_1 + norm_2 + epsilon) * diff
    loss_no_match = (norm_1 + norm_2) * diff
    loss = tf.reduce_sum(label * loss_match + (1-label) * loss_no_match)
    return loss

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm/(1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
    
def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False, name=None):
    with tf.name_scope(name, default_name="safe_2_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
        return tf.sqrt(squared_norm)

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
    
def placeholder_inputs(image_dims):
    """ Creates placeholders for siamese neural network.
    
    This method returns placeholders for inputs to the siamese network in both
    training, validation and testing. It also returns a placeholder for ground truth of 
    image pairs for training and validation.
    Input:
        image_dims - list of following structure: [height width 1] (the 1 is for grayscale images)
    Returns:
        image_holder - Placeholder with dynamic batch size and input image dimensions (192,192)    
        label_holder - ground truth labels
        handle - batch generator handler
    """
    
    left_image_holder = tf.placeholder(dtype=tf.float32, shape=[None, image_dims[2], image_dims[3], image_dims[-1]], name="left_image_holder") 
    right_image_holder = tf.placeholder(dtype=tf.float32, shape=[None, image_dims[2], image_dims[3], image_dims[-1]], name="right_image_holder") 
    label_holder = tf.placeholder(dtype=tf.float32, shape=[None], name="label_holder")
    handle = tf.placeholder(tf.string, shape=[],name="handle")
    
    return left_image_holder, right_image_holder, label_holder, handle
