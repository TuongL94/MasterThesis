#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:55:13 2018

@author: Tuong Lam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os

# imports from self-implemented modules
import utilities as util

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

def margin_loss(caps_input, gt, m_plus=0.9, m_minus=0.1, lambda_=0.5):
    caps_input_norms = safe_norm(caps_input, axis=-2, keepdims=True)
    present_errors = tf.square(tf.maximum(0.0, m_plus - caps_input_norms))
    present_errors = tf.reshape(present_errors, shape=[-1,10])
    absent_errors = tf.square(tf.maximum(0.0, caps_input_norms - m_minus))
    absent_errors = tf.reshape(absent_errors, shape=[-1,10])
    loss = tf.add(gt * present_errors, lambda_ * (1.0 - gt) * absent_errors)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss

def main(argv):
    tf.reset_default_graph()
    
    # Load mnist data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="X") # placeholder for input images
    batch_size = tf.shape(X)[0]
    
    caps1_n_maps = 32 
    caps1_n_caps = caps1_n_maps * 6 * 6
    caps1_n_dims = 8
    
    conv1 = tf.layers.conv2d(
        inputs = X,
        filters = 256,
        kernel_size = [9,9], 
        padding = "valid",
        activation = tf.nn.relu,
        name="conv1")
    
    conv2 = tf.layers.conv2d(
            inputs = conv1,
            filters = caps1_n_maps*caps1_n_dims,
            kernel_size = [9,9],
            strides = [2,2],
            padding = "valid",
            activation = tf.nn.relu,
            name = "conv2")
    
    caps1_raw = tf.reshape(conv2,[-1,caps1_n_caps,caps1_n_dims],name="caps1_raw")
    caps1_output = squash(caps1_raw, name="caps1_output")
    
    caps2_n_caps = 10
    caps2_n_dims = 16
    
    W_init = tf.random_normal(shape=[1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims],
                              stddev=0.1, dtype=tf.float32, name = "W_init")
    W = tf.Variable(W_init,trainable=True, name="W") # transformation matrices, will be trained with backpropagation
    W_tiled = tf.tile(W, [batch_size,1,1,1,1], name="W_tiled")
    
    # add additional dimensions to output of caps1 to conform with the dimensions of W_tiled
    caps1_output_expanded = tf.expand_dims(caps1_output, axis=-1, name="caps1_output_expanded")
    caps1_output_tile = tf.expand_dims(caps1_output_expanded, axis=2, name="caps1_output_tile")
    caps1_output_tiled = tf.tile(caps1_output_tile, [1,1,caps2_n_caps,1,1], name="caps1_output_tiled")
    
    caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")
    
    nbr_of_routing_iterations = 2
    counter = 1
    b_0 = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],dtype=tf.float32)
    
    def condition(b_coeff, counter):
        return tf.less(counter, 100)

    def loop_body(b_coeff, counter):
        routing_weights = tf.nn.softmax(b_coeff, axis=2, name="routing_weights")
        weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True, name="weighted_sum")
        v = squash(weighted_sum, axis=-2)
        v_tiled = tf.tile(v, [1, caps1_n_caps, 1, 1, 1], name="v_tiled")
        agreement = tf.matmul(caps2_predicted, v_tiled, transpose_a=True, name="agreement")
        new_b_coeff = tf.add(b_coeff, agreement)
        return [new_b_coeff,tf.add(1,counter)]
    
    b_final = tf.while_loop(condition, loop_body, [b_0, counter], maximum_iterations=nbr_of_routing_iterations)
        
    routing_weights = tf.nn.softmax(b_final[0], axis=2)
    weighted_predictions = tf.multiply(routing_weights, caps2_predicted)
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True)
    caps2_output = squash(weighted_sum, axis=-2)
        
    y_prob = safe_norm(caps2_output, axis=-2)
    y_prob_argmax = tf.argmax(y_prob, axis=2)
    y_pred = tf.squeeze(y_prob_argmax, axis=[1,2])
    
    label = tf.placeholder(shape=[None], dtype=tf.int64, name="label") # placeholder for ground truth labels
    
    gt = tf.one_hot(label, depth=caps2_n_caps)
    loss = margin_loss(caps2_output, gt)
    
    correct = tf.equal(label, y_pred, name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, name="train_op")
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    nbr_of_epochs = 10
    batch_size = 500
    
    n_iterations_per_epoch = mnist.train.num_examples // batch_size # training set size
    n_iterations_validation = mnist.validation.num_examples // batch_size # validation set size
    best_loss_val = np.infty # set validation loss to infinity
    output_dir = argv[0] # directory where the model is saved
    gpu_device_name = argv[1] # gpu device to use
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        if not os.path.exists(output_dir + "checkpoint"):
            print("No previous model exists, creating a new one.")
            init.run()
        else:
            saver.restore(sess, output_dir)
        
        with tf.device(gpu_device_name):
            # training loop    
            for i in range(nbr_of_epochs):
                for j in range(1, n_iterations_per_epoch + 1):
                    X_batch, label_batch = mnist.train.next_batch(batch_size)
                    X_batch = util.reshape_grayscale_data(X_batch)
                    _, train_loss_value = sess.run([train_op, loss], feed_dict={X:X_batch, label:label_batch})
                    print("\rIteration: {}/{} Loss: {:.5f}".format(j, n_iterations_per_epoch,train_loss_value), end="")
                
                loss_vals = []
                acc_vals = []
                for j in range(1, n_iterations_validation + 1):
                    X_batch, label_batch = mnist.validation.next_batch(batch_size)
                    validation_loss_value, acc_val = sess.run([loss, accuracy], feed_dict={X:util.reshape_grayscale_data(X_batch), label:label_batch})
                    loss_vals.append(validation_loss_value)
                    acc_vals.append(acc_val)
                    
                loss_val = np.mean(loss_vals)
                acc_val = np.mean(acc_vals)
                print("\rEpoch: {} Val accuracy: {:.4f}% Loss: {:.6f}{}".format(i+1, acc_val*100, loss_val, "(improved)" if loss_val < best_loss_val else ""))
                if loss_val < best_loss_val:
                    save_path = saver.save(sess, output_dir + "model")
                    print("Trained model saved in path: {}".format(save_path))
                    best_loss_val = loss_val
                
if __name__ == "__main__":
     main(sys.argv[1:])