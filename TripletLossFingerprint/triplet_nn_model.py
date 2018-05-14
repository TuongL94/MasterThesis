# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:55:36 2018

@author: Tuong Lam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def inference(input, training = True):
    seed = 1
    
    # Convolutional layer 1
    output = tf.layers.conv2d(
            inputs = input,
            filters = 16,
#            filters = 32,
            kernel_size = [7,7], 
#            strides = [2,2],
            padding = "valid",
            activation = tf.nn.relu,
            reuse = tf.AUTO_REUSE,
#                kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=4.0),
            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv1") 
    
#    output = tf.layers.dropout(
#            inputs = output,
#            rate = 0.4,
#            training = training,
#            seed = seed)
#    seed += 1
    
    # Pooling layer 1
    output = tf.layers.max_pooling2d(inputs = output, 
                                     pool_size = [2,2], 
                                     strides = 2)
    
    # Convolutional Layer 2 and pooling layer 2
    output = tf.layers.conv2d(
            inputs = output,
            filters = 16,
#            filters = 32,
            kernel_size = [5,5],
            padding = "same",
#            activation = tf.nn.relu,
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.initializers.truncated_normal(),
            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv2")
    
#    output = tf.layers.dropout(
#            inputs = output,
#            rate = 0.4,
#            training = training,
#            seed = seed)
#    seed += 1
            
    output = tf.layers.max_pooling2d(
            inputs = output, 
            pool_size = [2,2],
            strides = 2)
    
    # Convolutional Layer 3
    output = tf.layers.conv2d(
            inputs = output,
            filters = 32,
#            filters = 64,
            kernel_size = [3,3],
#            padding = "same",
#            activation = tf.nn.relu,
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv3")
    
#    output = tf.layers.dropout(
#            inputs = output,
#            rate = 0.4,
#            training = training,
#            seed = seed)
#    seed += 1
            
    output = tf.layers.max_pooling2d(
        inputs = output, 
        pool_size = [2,2],
        strides = 2)
    
    # Convolutional Layer 4
    output = tf.layers.conv2d(
            inputs = output,
            filters = 64,
#            filters = 128,
            kernel_size = [3,3],
#             padding = "same",
#            activation = tf.nn.relu,
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
#            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="conv4")
    
#    output = tf.layers.dropout(
#            inputs = output,
#            rate = 0.4,
#            training = training,
#            seed = seed)
#    seed += 1
            
    output = tf.layers.max_pooling2d(
            inputs = output, 
            pool_size = [2,2],
            strides = 2)
    

    output = tf.layers.flatten(output)
    
    output = tf.layers.dense(
            output,
            1000,
#            activation = tf.nn.relu,
            activation = tf.nn.leaky_relu,
            reuse = tf.AUTO_REUSE,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
            name="dense1")
    
#    output = tf.layers.dense(
#            output,
#            200,
#            activation = tf.nn.relu,
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
#            name="dense2")
#    
#    output = tf.layers.dropout(
#            inputs = output,
#            rate = 0.4,
#            training = training,
#            seed = seed)
#    seed += 1
    
#    output = tf.nn.l2_normalize(
#            output,
#            axis=1) 
    
    return output
    
    
    
#-------------- Batch Normalization -------------------------# 
#def inference(input, training = True):
#    
#    output = tf.layers.batch_normalization(
#            input,
#            training = training,
#            name = "batch_norm_1",
#            reuse = tf.AUTO_REUSE)
#    
#    # Convolutional layer 1
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 16,
#            kernel_size = [7,7],
#            strides = [2,2],
#            padding = "same",
#            reuse = tf.AUTO_REUSE,
##            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv1") 
#     
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_2",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 1)
#    
#    # Pooling layer 1
#    output = tf.layers.max_pooling2d(inputs = output, 
#                                     pool_size = [2,2], 
#                                     strides = 2)    
#    # Convolutional Layer 2
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 16,
#            kernel_size = [5,5],
#            strides = [2,2],
#            padding = "same",
#            reuse = tf.AUTO_REUSE,
##            kernel_initializer = tf.random_uniform_initializer(minval=-1, maxval=1),
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv2")
#        
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_3",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 2)
#    
#    # Pooling layer 2
#    output = tf.layers.max_pooling2d(
#            inputs = output, 
#            pool_size = [2,2],
#            strides = 2)
#    
#    # Convolutional Layer 3
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 32,
#            kernel_size = [3,3],
#            strides = [1,1],
#            padding = "same",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv3")
#            
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_4",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 3)
#    
#    # Convolutional Layer 4
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 64,
#            kernel_size = [3,3],
#            strides = [1,1],
#            padding = "same",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv4")
#        
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_5",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 4)
#    
#    output = tf.layers.flatten(
#            output)
#    
#    output = tf.layers.dense(
#        output,
#        1024,
#        reuse = tf.AUTO_REUSE,
#        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#        name="dense")
#    
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_6",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#        
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 5)
#    
#    output = tf.nn.l2_normalize(
#            output,
#            axis=1)
#              
#    return output
    

#--------------- Deep---------------------------#

#def inference(input, training = True):
#    
#    output = tf.layers.batch_normalization(
#            input,
#            training = training,
#            name = "batch_norm_1",
#            reuse = tf.AUTO_REUSE)
#    
#    # Convolutional layer 1
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 16,
#            kernel_size = [9,9],
#            strides = [1,1],
#            padding = "valid",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv1") 
#     
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_2",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 1)
#    
#    # Pooling layer 1
#    output = tf.layers.max_pooling2d(inputs = output, 
#                                     pool_size = [2,2], 
#                                     strides = 2)    
#    # Convolutional Layer 2
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 16,
#            kernel_size = [5,5],
#            strides = [1,1],
#            padding = "valid",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv2")
#        
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_3",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 2)
#    
#    # Pooling layer 2
#    output = tf.layers.max_pooling2d(
#            inputs = output, 
#            pool_size = [2,2],
#            strides = 2)
#    
#    # Convolutional Layer 3
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 32,
#            kernel_size = [3,3],
#            strides = [1,1],
#            padding = "same",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv3")
#    
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_4",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 3)
#    
#    # Convolutional Layer 4
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 64,
#            kernel_size = [3,3],
#            strides = [1,1],
#            padding = "valid",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv4")
#    
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_5",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 4)
#    
#    # Convolutional Layer 5
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 64,
#            kernel_size = [3,3],
#            padding = "valid",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv5")
#    
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_6",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 5)
#    
#    # Convolutional Layer 6
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 128,
#            kernel_size = [3,3],
#            padding = "valid",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv6")
#    
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_7",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 6)
#    
#    output = tf.layers.max_pooling2d(inputs = output, 
#                                     pool_size = [2,2], 
#                                     strides = 2)  
#    
#    # Convolutional Layer 6
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 128,
#            kernel_size = [1,1],
#            padding = "valid",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv7")
#    
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_8",
#        reuse = tf.AUTO_REUSE)
#    
#    # Convolutional Layer 6
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 256,
#            kernel_size = [1,1],
#            padding = "valid",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv8")
#    
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_9",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 7)
#    
#    ############  NEW  #################
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 256,
#            kernel_size = [3,3],
#            padding = "valid",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv_extra_1")
#    
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_new_1",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 7)
#    
#    output = tf.layers.max_pooling2d(inputs = output, 
#                                     pool_size = [2,2], 
#                                     strides = 2)  
#    
#    output = tf.layers.conv2d(
#            inputs = output,
#            filters = 256,
#            kernel_size = [3,3],
#            padding = "valid",
#            reuse = tf.AUTO_REUSE,
#            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#            name="conv_extra_3")
#    
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_new_2",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 8)
#    #-----------------------------------
#    
#    output = tf.layers.flatten(
#            output)
#    
#    #############  NEW  ####################
#    output = tf.layers.dense(
#        output,
#        4096,
#        reuse = tf.AUTO_REUSE,
#        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#        name="dense1")
#    
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_hidden_dense1",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
#    output = tf.layers.dropout(
#        output,
#        rate = 0.5,
#        training = training,
#        seed = 9)
#    #---------------------------------------
#    
#    output = tf.layers.dense(
#        output,
#        1024,
#        reuse = tf.AUTO_REUSE,
#        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.3),
#        name="dense2")
#    
#    output = tf.layers.batch_normalization(
#        output,
#        training = training,
#        name = "batch_norm_10",
#        reuse = tf.AUTO_REUSE)
#    
#    output = tf.nn.leaky_relu(
#            output)
#    
##    output = tf.layers.dropout(
##        output,
##        rate = 0.5,
##        training = training,
##        seed = 10)
#    
#    output = tf.nn.l2_normalize(
#            output,
#            axis=1)
#              
#    return output
    


    
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

def triplet_loss(anchor, positive, negative, margin):
    """ Computes the contrastive loss between two vectors
    
    Input:
        anchor - output from network when input is the anchor image
        positive - output from network when input is the positve image
        negative - output from network when input is the negative image
        margin - margin for triplet loss, positive constant
    Returns:
        loss - the triplet loss 
    """
    distance_pos = tf.reduce_sum(tf.square(anchor - positive),1)
    distance_neg = tf.reduce_sum(tf.square(anchor - negative),1)
    
    loss = tf.maximum(0., margin + distance_pos - distance_neg)
    loss = tf.reduce_mean(loss)
    return loss

def training(loss, learning_rate, momentum):
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum, use_nesterov=True)
    train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op
    
def placeholder_inputs(image_dims,batch_sizes):
    """ Creates placeholders for triplet neural network.
    
    This method returns placeholders for inputs to the triplet network in both
    training, validation and testing. It also returns a placeholder for ground truth of 
    image pairs for training and validation.
    Input:
    image_dims - list of following structure: [height width 1] (the 1 is for grayscale images)
    batch_sizes - list of batch sizes for training,validation and testing respectively 
    Returns:
    anchor_train - placeholder for anchor input to triplet network for training
    positive_train - placeholder for positive input to triplet network for training
    label_train - placeholder for ground truths of image pairs for training
    anchor_val - placeholder for anchor input to triplet network for validation
    positive_val - placeholder for positive input to triplet network for validation
    label_val - placeholder for ground truths of image pairs for validation
    anchor_test - placeholder for anchor input to triplet network for testing
    positive_test - placeholder for positive input to triplet network for testing
    """
#    anchor_train = tf.placeholder(tf.float32, [batch_sizes[0],image_dims[0],image_dims[1],image_dims[2]], name="anchor_train")
#    positive_train = tf.placeholder(tf.float32,[batch_sizes[0],image_dims[0],image_dims[1],image_dims[2]], name="positive_train")
#    negative_train = tf.placeholder(tf.float32,[batch_sizes[0],image_dims[0],image_dims[1],image_dims[2]], name="negative_train")
##    label_train = tf.placeholder(tf.float32, [batch_sizes[0],1], name="label_train") # 1 if same, 0 if different
#    
#    anchor_val = tf.placeholder(tf.float32,[batch_sizes[1],image_dims[0],image_dims[1],image_dims[2]], name="anchor_val")
#    positive_val = tf.placeholder(tf.float32, [batch_sizes[1],image_dims[0],image_dims[1],image_dims[2]], name="positive_val")
#    negative_val = tf.placeholder(tf.float32,[batch_sizes[0],image_dims[0],image_dims[1],image_dims[2]], name="negative_val")
##    label_val = tf.placeholder(tf.float32, [batch_sizes[1],1], name="label_val") # 1 if same, 0 if different
#    
#    left_test = tf.placeholder(tf.float32, [batch_sizes[2],image_dims[0],image_dims[1],image_dims[2]], name="left_test")
#    right_test = tf.placeholder(tf.float32, [batch_sizes[2],image_dims[0],image_dims[1],image_dims[2]], name="right_test")
    
    anchor_train = tf.placeholder(tf.float32, [None,image_dims[0],image_dims[1],image_dims[2]], name="anchor_train")
    positive_train = tf.placeholder(tf.float32,[None,image_dims[0],image_dims[1],image_dims[2]], name="positive_train")
    negative_train = tf.placeholder(tf.float32,[None,image_dims[0],image_dims[1],image_dims[2]], name="negative_train")
#    label_train = tf.placeholder(tf.float32, [batch_sizes[0],1], name="label_train") # 1 if same, 0 if different
    
    anchor_val = tf.placeholder(tf.float32,[None,image_dims[0],image_dims[1],image_dims[2]], name="anchor_val")
    positive_val = tf.placeholder(tf.float32, [None,image_dims[0],image_dims[1],image_dims[2]], name="positive_val")
    negative_val = tf.placeholder(tf.float32,[None,image_dims[0],image_dims[1],image_dims[2]], name="negative_val")
#    label_val = tf.placeholder(tf.float32, [batch_sizes[1],1], name="label_val") # 1 if same, 0 if different
    
    left_test = tf.placeholder(tf.float32, [None,image_dims[0],image_dims[1],image_dims[2]], name="left_test")
    right_test = tf.placeholder(tf.float32, [None,image_dims[0],image_dims[1],image_dims[2]], name="right_test")
    
#    return anchor_train,positive_train,label_train,anchor_val,positive_val,label_val,anchor_test,positive_test
    return anchor_train,positive_train,negative_train,anchor_val,positive_val,negative_val,left_test,right_test
    