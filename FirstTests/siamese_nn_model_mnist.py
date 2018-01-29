# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:55:36 2018

@author: Tuong Lam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_generator import data_generator
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

import numpy as np
import scipy.linalg as sl
import tensorflow as tf

def inference(input, reuse=True):
    input_layer = input
    
    # Convolutional layer 1
    conv1 = tf.layers.conv2d(
            inputs = input_layer,
            filters = 32,
            kernel_size = [5, 5], 
            padding = "same",
            activation = tf.nn.relu)
    
    # Pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, 
                                     pool_size = [2,2], 
                                     strides = 2)
    
    # Convolutional Layer 2 and pooling layer 2
    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 64,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
            
    pool2 = tf.layers.max_pooling2d(
            inputs = conv2, 
            pool_size = [2,2],
            strides = 2)
    
    net = tf.contrib.layers.flatten(pool2)
    return net
    
    
def loss(input_1,input_2):
    return tf.linalg.norm([input_1,input_2])

def training(loss, learning_rate, momentum):
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum, use_nesterov=True)
    train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op
    
def placeholder_inputs(batch_size):
    left = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name='left')
    right = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name='right')
    label = tf.placeholder(tf.int32, [batch_size, 1], name='label') # 1 if same, 0 if different
    return left,right,label
    
         
def main(unused_argv):
    
    #Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    batch_size = 5
    train_iter = 2000
    
    generator = data_generator(train_data,train_labels) # initialize data generator
    left,right,label = placeholder_inputs(batch_size) # create placeholders for pairs of images and ground truth matching
    left_output = inference(left, reuse=False)
    right_output = inference(right, reuse=True)
    l2_loss = loss(left_output, right_output)
    
    learning_rate = tf.constant(0.01, name = 'learning_rate')
    momentum = tf.constant(0.99, name = 'momentum')
    train_op = training(l2_loss, learning_rate, momentum)

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # initialize all trainable parameters
        
        for i in range(train_iter):
            b_l, b_r, b_sim = generator.gen_batch(batch_size)
            _,loss_value = sess.run([train_op, l2_loss],feed_dict={left:b_l, right:b_r, label:b_sim})
            if i % 100 == 0:
                print('Iteration %d: loss = %.2f' % (i, loss_value))
                
    
    
    
if __name__ == "__main__":
    tf.app.run()