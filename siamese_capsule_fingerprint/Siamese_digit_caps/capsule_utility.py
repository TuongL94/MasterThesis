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


#### ------------------------- Loss functions -------------------------- ####

#def contrastive_loss(input_1,input_2,label,margin):
#    """ Computes the contrastive loss between two vectors
#    
#    Input:
#    input_1 - first input vector
#    input_2 - second input vector
#    label - ground truth for similarity between the vectors. 1 if they are similar, 0 otherwise.
#    margin - margin for contrastive loss, positive constant
#    Returns the contrastive loss between input_1 and input_2 with specified margin.
#    """
#    d_sq = tf.reduce_sum(tf.pow(input_1-input_2,2),1,keepdims=True)
#    max_sq = tf.square(tf.maximum(margin-d_sq,0))
#    return tf.reduce_mean(label*d_sq + (1-label)*max_sq)/2

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
    diff = safe_norm(input_1 - input_2, axis=-2)
    loss_match = tf.reduce_sum(tf.truediv(1.0,norm_1 + norm_2 + epsilon) * diff, axis=2)
    loss_no_match = tf.reduce_sum((norm_1 + norm_2) * diff, axis=2)
    loss = tf.reduce_mean(label * loss_match + (1-label) * loss_no_match)
    return loss

def triplet_caps_loss(anchor, pos, neg, margin):   
    dist_pos = tf.reduce_sum(tf.reduce_sum(tf.square(anchor - pos), axis=-2), axis=-2)  # Change axis depending on which typ of CapsNet (Check dimensions on output)
    dist_neg = tf.reduce_sum(tf.reduce_sum(tf.square(anchor- neg), axis=-2), axis=-2)
    loss = tf.maximum(0., margin + dist_pos - dist_neg)
    loss = tf.reduce_mean(loss)
    return loss

def contrastive_caps_loss(input_1, input_2, label, margin):
#    dist = tf.reduce_sum(tf.reduce_sum(tf.square(input_1 - input_2), axis=-2), axis=-2)
    dist = tf.reduce_sum(safe_norm(input_1 - input_2, axis=-2, epsilon=0), axis=-2)
    margin_max = tf.square(tf.maximum(0., margin - dist))
    loss = tf.reduce_mean(label*tf.square(dist) + (1-label) * margin_max) / 2
    return loss

#####------------------ Agreement Loss -------------------######
    
def agreement_loss(input_1, input_2, label, active_threshold, inactive_threshold, margin, epsilon = 1e-7):
    # Set part minimal part of features vectors to consider as active/inactive
    
    
    # Squeeze input
    input_1 = tf.squeeze(input_1)
    input_2 = tf.squeeze(input_2)
    
    # Calculate length of feature vectors
    feature_length_1 = safe_norm(input_1, axis=-1, keepdims=True, name="feature_length_1")    # 2-norm  dims: [batch_size, 100, 1]
    feature_length_2 = safe_norm(input_2, axis=-1, keepdims=True, name="feature_length_2")    # 2-norm
    
    # Create active vectors
    active_1 = tf.maximum(0., feature_length_1 - active_threshold, name="active_1")
    active_2 = tf.maximum(0., feature_length_2 - active_threshold, name="active_2")
    # Calculate elementwise multiplication between images active vectors
    active = tf.multiply(active_1, active_2, name="active")
    
    
    # Create inactive vectors
    inactive_1 = tf.maximum(0., inactive_threshold - feature_length_1, name="inactive_1")
    inactive_2 = tf.maximum(0., inactive_threshold - feature_length_2, name="inactive_2")
    # Calculate elementwise multiplication between images inactive vectors
    inactive = tf.multiply(inactive_1, inactive_2, name="inactive")
    
    # Calculate agreement measure (elements wise muliplication between active and inactive vector)
    agreement = tf.multiply(active, inactive, name="agreement")
    
    
    # Calculate angles between feature vectors (calculate dot product between normalized feature vectors)
    # Normalize feature vectors
    feature_1 = tf.div(input_1, feature_length_1,name="normalize_1")   #dims: [batch_size, 100, 8]
    feature_2 = tf.div(input_2, feature_length_2,name="normalize_2")
    # Calculate dot product
    # This is using the range [1,-1] symmetricly (i.e. cos()) could pose problem. Has advantage of not having the edge 360 -> 0
    dot_product = tf.matmul(tf.expand_dims(feature_1,-1), tf.expand_dims(feature_2,-1), transpose_a = True, name="dot_product")     # dims: [batch_size, 100, 1, 1]
    
    # Calculate weighted mean
    weighted_dot_product = tf.reduce_sum(tf.multiply(dot_product, tf.expand_dims(active,-1)), axis=1, name="weighted_dot_product")   # (dims: [batch_size, 1, 1])
    weighted_mean = tf.div(weighted_dot_product, tf.expand_dims(tf.reduce_sum(active, axis = 1) + epsilon, -1), name = "weighted_mean")   # (dims: [batch_size,])
    
    # Calculate weighted variance using weighted mean
    divergance = tf.squeeze(dot_product, axis = -1) - weighted_mean
    weighted_variance_numerator = tf.reduce_sum(tf.multiply(active, tf.multiply(divergance, divergance)), axis = 1, name="weighted_variance_numerator")
    W1 = tf.reduce_sum(active, axis = 1, name="W1")
    W2 = tf.reduce_sum(tf.multiply(active, active), axis = 1, name="W2")
    
    weighted_variance = tf.div(weighted_variance_numerator, W1 - tf.div(W2, W1 + epsilon) + epsilon)
    
    # Scale weighted variance with sum of agreement
    variance_loss = tf.squeeze(tf.div(weighted_variance, tf.reduce_sum(agreement, axis = 1) + epsilon), name = "varaince_loss")
    
    # Use ground truth labels to output correct loss (like contrastive loss)
    loss_match = tf.multiply(label, variance_loss, name="loss_match")
    loss_non_match = (1-label) * tf.maximum(0., margin - variance_loss, name="loss_non_match")
    loss = tf.reduce_mean(loss_match + loss_non_match, name="loss")
    return loss


################################################################

#####------------------ Reconstruction loss --------------------######
    
def reconstruction_loss(left_image, right_image, recon_left, recon_right, alpha):
    image_dims = right_image.get_shape()
    left_image_flat = tf.reshape(left_image, [-1, image_dims[1] * image_dims[2]], name = "flat_left_image")
    right_image_flat = tf.reshape(right_image, [-1, image_dims[1] * image_dims[2]], name = "flat_right_image")
    
    squared_diff_left = tf.square(left_image_flat - recon_left, name="squared_diff_left")
    squared_diff_right= tf.square(right_image_flat - recon_left, name="squared_diff_right")
    
    reconstruct_loss = alpha * tf.reduce_mean(squared_diff_left + squared_diff_right, name = "reconstruction_loss")
    return reconstruct_loss


###################################################################################

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm/(1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
    
def safe_norm(s, axis=-1, epsilon=1e-9, keepdims=False, name=None):
    with tf.name_scope(name, default_name="safe_2_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
        return tf.sqrt(squared_norm + epsilon)

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
