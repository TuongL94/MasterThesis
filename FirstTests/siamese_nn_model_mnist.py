# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:55:36 2018

@author: Tuong Lam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_generator import data_generator


import numpy as np
import scipy.linalg as sl
import tensorflow as tf

def siamese_model_fn(input, reuse=False):
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
    
def matnorm_loss(input_1,input_2):
    return sl.norm(input_1,input_2)
    
def main(unused_argv):
    
    #Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    batch_size = 100
    train_iter = 2000
    steps =500
    
    generator = data_generator(train_data,train_labels)
    train_left,train_right,gt = generator.gen_batch(batch_size)
    print(np.shape(train_left))
 
#    
#    left = tf.placeholder(tf.float32, [None, 28, 28, 1], name='left')
#    right = tf.placeholder(tf.float32, [None, 28, 28, 1], name='right')
#   
#    left_output = mynet(left, reuse=False)
#
#    right_output = mynet(right, reuse=True)
#
#    loss = matnorm_loss(left_output, right_output)
#    
#
if __name__ == "__main__":
    tf.app.run()