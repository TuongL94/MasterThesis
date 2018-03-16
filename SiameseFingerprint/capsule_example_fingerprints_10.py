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
import pickle
import matplotlib.pyplot as plt
import time

# imports from self-implemented modules
import utilities as util
from data_generator import data_generator

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

def margin_loss(caps_input, gt, nbr_of_classes, m_plus=0.9, m_minus=0.1, lambda_=0.5):
    caps_input_norms = safe_norm(caps_input, axis=-2, keepdims=True)
    present_errors = tf.square(tf.maximum(0.0, m_plus - caps_input_norms))
    present_errors = tf.reshape(present_errors, shape=[-1,nbr_of_classes])
    absent_errors = tf.square(tf.maximum(0.0, caps_input_norms - m_minus))
    absent_errors = tf.reshape(absent_errors, shape=[-1,nbr_of_classes])
    loss = tf.add(gt * present_errors, lambda_ * (1.0 - gt) * absent_errors)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss

def get_classification_data(generator, nbr_of_classes):
    
    breakpoints = generator.breakpoints_train
    matches = generator.match_train
    index_list = [np.array([],dtype=np.int64) for i in range(len(breakpoints)-1)]
    image_dims = np.shape(generator.train_data[0])
    images = np.zeros((0,image_dims[1],image_dims[1],image_dims[-1]))
    gt = np.zeros(0)
    counter = 1
    
    for i in range(1,len(breakpoints)):
        for j in range(breakpoints[i-1], breakpoints[i]):
            ind = np.where(matches == j)
            if np.shape(ind[0])[0] > 0:
                index_list[i-1] = np.append(index_list[i-1],j)
    
    for e in index_list:
        e = np.unique(e)
        
    index_list.sort(key = lambda x: len(x))
    for i in range(len(index_list)-1,len(index_list)-nbr_of_classes-1,-1):
        images = np.append(images, np.take(generator.train_data[0],index_list[i],axis=0), axis=0)
        gt = np.append(gt,counter*np.ones(len(index_list[i])))
        counter += 1
        
    # Pick out 50 random fingerprints to use as a non matching set   
    nbr_non_matching = 50
    random_finger = np.random.randint(0,len(index_list)-nbr_of_classes)
    random_images_idx = index_list[random_finger][np.random.randint(0, len(index_list[random_finger]))]
    for i in range(nbr_non_matching - 1):
        random_finger = np.random.randint(0,len(index_list)-nbr_of_classes)
        random_images_idx = np.append(random_images_idx, index_list[random_finger][np.random.randint(0, len(index_list[random_finger]))])
    
    # Add the random fingerprints to the matching
    images = np.append(images, np.take(generator.train_data[0], random_images_idx, axis=0), axis=0)
    gt = np.append(gt, counter*np.ones(nbr_non_matching))
        
    return images,gt

def main(argv):
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir = argv[0] # directory where the model is saved
    gpu_device_name = argv[1] # gpu device to use
    
    if len(argv) == 3:
        use_time = True
    else:
        use_time = False
    
    # Load fingerprint data and create a data_generator instance if one 
    # does not exist, otherwise load existing data_generator
    if not os.path.exists(dir_path + "/generator_data.pk1"):
        with open('generator_data.pk1', 'wb') as output:
            # Load fingerprint label_holders and data from file with names
            
            finger_id = np.load(dir_path + "/finger_id_mt_vt_112.npy")
            person_id = np.load(dir_path + "/person_id_mt_vt_112.npy")
            finger_data = np.load(dir_path + "/fingerprints_mt_vt_112.npy")
            translation = np.load(dir_path + "/translation_mt_vt_112.npy")
            rotation = np.load(dir_path + "/rotation_mt_vt_112.npy")
            finger_data = util.reshape_grayscale_data(finger_data)
            nbr_of_images = np.shape(finger_data)[0] # number of images to use from the original data set
            
            rotation_res = 1
            generator = data_generator(finger_data, finger_id, person_id, translation, rotation, nbr_of_images, rotation_res) # initialize data generator
            
            finger_id_gt_vt = np.load(dir_path + "/finger_id_gt_vt.npy")
            person_id_gt_vt = np.load(dir_path + "/person_id_gt_vt.npy")
            finger_data_gt_vt = np.load(dir_path + "/fingerprints_gt_vt.npy")
            translation_gt_vt = np.load(dir_path + "/translation_gt_vt.npy")
            rotation_gt_vt = np.load(dir_path + "/rotation_gt_vt.npy")
            finger_data_gt_vt = util.reshape_grayscale_data(finger_data_gt_vt)
            nbr_of_images_gt_vt = np.shape(finger_data_gt_vt)[0]
            
            generator.add_new_data(finger_data_gt_vt, finger_id_gt_vt, person_id_gt_vt, translation_gt_vt, rotation_gt_vt, nbr_of_images_gt_vt)
            pickle.dump(generator, output, pickle.HIGHEST_PROTOCOL)
    else:
        # Load generator
        with open('generator_data.pk1', 'rb') as input:
            generator = pickle.load(input)
    
    # Set number of 
    nbr_of_classes = 21
    
    train_images, train_gt = get_classification_data(generator,nbr_of_classes)  # OBS!!! adding random fingerprints in last class
    image_dims = np.shape(train_images)
    
    # parameters for training
    batch_size_train = 3
    train_itr = 2000000
    
    save_itr = 10000 # frequency in which the model is saved

    tf.reset_default_graph()
    
    with tf.device(gpu_device_name):
    # create placeholders
        image_holder = tf.placeholder(dtype=tf.float32, shape=[None, image_dims[1], image_dims[1], image_dims[-1]], name="image_holder") 
        label_holder = tf.placeholder(dtype=tf.int64, shape=[None], name="label_holder")
        handle = tf.placeholder(tf.string, shape=[],name="handle")
        
        # Setup tensorflow's batch generator
        train_match_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_gt))
        train_match_dataset = train_match_dataset.shuffle(buffer_size=np.shape(train_images)[0])
        train_match_dataset = train_match_dataset.repeat()
        train_match_dataset = train_match_dataset.batch(batch_size_train)
        
        train_iterator = train_match_dataset.make_one_shot_iterator()
        iterator = tf.data.Iterator.from_string_handle(handle, train_match_dataset.output_types)
        next_element = iterator.get_next()
    
    caps1_n_maps = 32 
    caps1_n_caps = caps1_n_maps * 42 * 42
    caps1_n_dims = 8
    
    with tf.device(gpu_device_name):
        conv1 = tf.layers.conv2d(
            inputs = image_holder,
            filters = 256,
            kernel_size = [9,9], 
            strides = [2,2],
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
        
        caps2_n_caps = nbr_of_classes
        caps2_n_dims = 16
        
        W_init = tf.random_normal(shape=[1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims],
                                  stddev=0.1, dtype=tf.float32, name = "W_init")
        W = tf.Variable(W_init,trainable=True, name="W") # transformation matrices, will be trained with backpropagation
        W_tiled = tf.tile(W, [batch_size_train,1,1,1,1], name="W_tiled")
        
        # add additional dimensions to output of caps1 to conform with the dimensions of W_tiled
        caps1_output_expanded = tf.expand_dims(caps1_output, axis=-1, name="caps1_output_expanded")
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, axis=2, name="caps1_output_tile")
        caps1_output_tiled = tf.tile(caps1_output_tile, [1,1,caps2_n_caps,1,1], name="caps1_output_tiled")
        
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")
        
        nbr_of_routing_iterations = 2
        counter = 1
        b_0 = tf.zeros([batch_size_train, caps1_n_caps, caps2_n_caps, 1, 1],dtype=tf.float32)
    
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
        
        gt = tf.one_hot(label_holder, depth=caps2_n_caps)
        loss = margin_loss(caps2_output, gt, nbr_of_classes)
        
        correct = tf.equal(label_holder, y_pred, name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, name="train_op")
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        with tf.device(gpu_device_name):
#            if not os.path.exists(output_dir + "checkpoint"):
            print("No previous model exists, creating a new one.")
            init.run()
#            else:
#                saver.restore(sess, output_dir)

            train_match_handle = sess.run(train_iterator.string_handle())
            
            graph = tf.get_default_graph()
            # Summary setup
            
            # get parameters of the first convolutional layer. Add filters and histograms
            # of filters and biases to summary
            conv1_filters = graph.get_tensor_by_name("conv1/kernel:0")
            nbr_of_filters_conv1 = sess.run(tf.shape(conv1_filters)[-1])
            hist_conv1 = tf.summary.histogram("hist_conv1", conv1_filters)
            conv1_filters = tf.transpose(conv1_filters, perm = [3,0,1,2])
            filter1 = tf.summary.image('Filter_1', conv1_filters, max_outputs=nbr_of_filters_conv1)
            conv1_bias = graph.get_tensor_by_name("conv1/bias:0")
            hist_bias1 = tf.summary.histogram("hist_bias1", conv1_bias)

            conv2_filters = graph.get_tensor_by_name("conv2/kernel:0")
            hist_conv2 = tf.summary.histogram("hist_conv2", conv2_filters)
            conv2_bias = graph.get_tensor_by_name("conv2/bias:0")
            hist_bias2 = tf.summary.histogram("hist_bias2", conv2_bias)
            
            # transpose filters to coincide with the dimensions requested by tensorflow's summary. 
            # Add filters to summary
#            conv1_filters = tf.transpose(conv1_filters, perm = [3,0,1,2])
#            filter1 = tf.summary.image('Filter_1', conv1_filters, max_outputs=nbr_of_filters_conv1)

            summary_train_loss = tf.summary.scalar('training_loss', loss)
            
            summary_op = tf.summary.merge([summary_train_loss, filter1, hist_conv1, hist_bias1, hist_conv2, hist_bias2])
            train_writer = tf.summary.FileWriter(output_dir + "train_summary", graph=tf.get_default_graph())
            
            average_acc = []
            average_loss = []
            start_time_train = time.time()
            # training loop    
            for i in range(1, train_itr + 1):
                image_batch, gt_batch = sess.run(next_element,feed_dict={handle:train_match_handle})
                _, train_loss_value, acc_val, summary = sess.run([train_op, loss, accuracy, summary_op], feed_dict={image_holder:image_batch, label_holder:gt_batch})
#                if i % 50 == 0:
#                    print("\rIteration: {} Loss: {:.5f} Accuracy: {:.5f}".format(i,train_loss_value,acc_val))
                    
                if i % save_itr == 0 or i == train_itr:
                    save_path = tf.train.Saver().save(sess,output_dir + "model")
                    print("Trained model after {} iterations saved in path: {}".format(i,save_path))
                    
                if i % 500 == 0 or i == train_itr:
                    acc_val_total = 0
                    test_loss_total = 0
                    for j in range(train_images.shape[0] // batch_size_train):
                        image_batch, gt_batch = sess.run(next_element,feed_dict={handle:train_match_handle})
                        acc_val, test_loss = sess.run([accuracy, loss], feed_dict={image_holder:image_batch, label_holder:gt_batch})
                        acc_val_total += acc_val
                        test_loss_total += test_loss
                    average_acc.append(acc_val_total/(train_images.shape[0]//batch_size_train)  )
                    average_loss.append(test_loss_total/(train_images.shape[0]//batch_size_train))
                    print("Average accuracy on training set after %d iterations: %f" % (i,average_acc[-1]))
                    print("Average loss on trainning set after %d iterationds: %f" % (i,average_loss[-1]))
                    
                    if use_time:
                        elapsed_time = (time.time() - start_time_train)/60.0 # elapsed time in minutes since start of training 
                        if elapsed_time >= int(argv[2]):
                            save_path = tf.train.Saver().save(sess,output_dir + "model")
                            print("Trained model after {} iterations saved in path: {}".format(i,save_path))
                            break
                    
                train_writer.add_summary(summary, i)
                
        # Plot precision over time
        time_points = list(range(len(average_acc)))
        plt.plot(time_points, average_acc)
        plt.title("Average accuracy over time")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.show()
    
        if len(average_acc) > 0:
            print("Final (last computed) accuracy: %f" % average_acc[-1])
        
        # Plot validation loss over time
        plt.figure()
        plt.plot(time_points, average_loss)
        plt.title("Average loss over time")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
                
if __name__ == "__main__":
     main(sys.argv[1:])