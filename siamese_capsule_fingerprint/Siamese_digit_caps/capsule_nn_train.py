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
import re
import time
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt

#sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "../SiameseFingerprint/utilities.py")

# imports from self-implemented modules
import utilities as util
from data_generator import data_generator
import capsule_utility as cu
import capsule_nn_eval as ce
import capsule_nn_model as cm

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
    return images,gt

def main(argv):
    
    output_dir = argv[0]
    data_path =  argv[1]
    gpu_device_name = argv[2]
    if len(argv) == 4:
        use_time = True
    else:
        use_time = False

    # Load fingerprint data and create a data_generator instance if one 
    # does not exist, otherwise load existing data_generator
    if not os.path.exists(data_path + "generator_data.pk1"):
        with open(data_path + "generator_data.pk1", "wb") as output:
            # Load fingerprint labels and data from file with names
            finger_id = np.load(data_path + "finger_id.npy")
            person_id = np.load(data_path + "person_id.npy")
            finger_data = np.load(data_path + "fingerprints.npy")
            translation = np.load(data_path + "translation.npy")
            rotation = np.load(data_path + "rotation.npy")
            finger_data = util.reshape_grayscale_data(finger_data)
            nbr_of_images = np.shape(finger_data)[0] # number of images to use from the original data set
            
            rotation_res = 1
            generator = data_generator(finger_data, finger_id, person_id, translation, rotation, nbr_of_images, rotation_res) # initialize data generator
            
            finger_id_gt_vt = np.load(data_path + "finger_id_gt_vt.npy")
            person_id_gt_vt = np.load(data_path + "person_id_gt_vt.npy")
            finger_data_gt_vt = np.load(data_path + "fingerprints_gt_vt.npy")
            translation_gt_vt = np.load(data_path + "translation_gt_vt.npy")
            rotation_gt_vt = np.load(data_path + "rotation_gt_vt.npy")
            finger_data_gt_vt = util.reshape_grayscale_data(finger_data_gt_vt)
            nbr_of_images_gt_vt = np.shape(finger_data_gt_vt)[0]
            
            generator.add_new_data(finger_data_gt_vt, finger_id_gt_vt, person_id_gt_vt, translation_gt_vt, rotation_gt_vt, nbr_of_images_gt_vt)
            pickle.dump(generator, output, pickle.HIGHEST_PROTOCOL)
    else:
        # Load generator
        with open(data_path + "generator_data.pk1", 'rb') as input:
            generator = pickle.load(input)
    
#    train_images, train_gt = get_classification_data(generator,10)
    image_dims = np.shape(generator.train_data)
    
    # parameters for training
    batch_size_train = 500    # OBS! Has to be multiple of 2
    train_itr = 500000000
    
    learning_rate = 0.000001
    momentum = 0.99
    
    # Hyper parameters
    routing_iterations = 2
    digit_caps_classes = 20
    digit_caps_dims = 8
    
    caps1_n_maps = 16 
    caps1_n_dims = 8
    
    # Paramters for validation set
    batch_size_val = 500
    val_itr = 33
    threshold = 0.0001
    thresh_step = 0.00001
    nbr_val_itr = 3
    
    save_itr = 250 # frequency in which the model is saved

    tf.reset_default_graph()
    
    if not os.path.exists(output_dir + "checkpoint"):
        print("No previous model exists, creating a new one.")
        is_model_new = True
        meta_file_exists = False
        current_itr = 0 # current training iteration
        
        with tf.device(gpu_device_name):
            # create placeholders
            left_image_holder, right_image_holder, label_holder, handle = cu.placeholder_inputs(image_dims)
                
            left_train_output = cm.capsule_net(left_image_holder, routing_iterations, digit_caps_classes, digit_caps_dims, 
                                               caps1_n_maps, caps1_n_dims, batch_size_train, name="left_train")
            right_train_output = cm.capsule_net(right_image_holder, routing_iterations, digit_caps_classes, digit_caps_dims,
                                                caps1_n_maps, caps1_n_dims, batch_size_train, name="right_train")
            
#            anchor_val_output = sm.inference(anchor_val)
#            pos_val_output = sm.inference(pos_val)
#            
#            left_test_output = sm.inference(left_test)
#            right_test_output = sm.inference(right_test)
            
#            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#            margin = tf.constant(4.0) # margin for contrastive loss
#            train_loss = sm.triplet_loss(anchor_train_output,pos_train_output,neg_train_output,margin)
#            # add regularization terms to triplet loss function
#            for i in range(len(reg_losses)):
#                train_loss += reg_losses[i]
            
#            val_loss = sm.triplet_loss(anchor_val_output,pos_val_output,neg_val_output,margin)
            
            train_loss = cu.scaled_pair_loss(left_train_output, right_train_output, label_holder)
            
            tf.add_to_collection("train_loss",train_loss)
#            tf.add_to_collection("val_loss",val_loss)
            tf.add_to_collection("left_train_output",left_train_output)
            tf.add_to_collection("right_train_output",right_train_output)
#            tf.add_to_collection("neg_val_output",neg_val_output)
#            tf.add_to_collection("left_test_output",left_test_output)
#            tf.add_to_collection("right_test_output",right_test_output)
            
#            global_vars = tf.global_variables()
#            for i in range(len(global_vars)):
#                print(global_vars[i])
                
            saver = tf.train.Saver()

    else:
        print("Using latest existing model in the directory " + output_dir)
        is_model_new = False
        meta_file_exists = True
         
        with open(output_dir + "checkpoint","r") as file:
            line  = file.readline()
            words = re.split("/",line)
            model_file_name = words[-1][:-2]
            current_itr = int(re.split("-",model_file_name)[-1]) # current training iteration
            for file in os.listdir(output_dir):
                if file.endswith(".meta"):
                    meta_file_name = os.path.join(output_dir,file)
            saver = tf.train.import_meta_graph(meta_file_name)
        
        with tf.device(gpu_device_name):
            g = tf.get_default_graph()
            left_image_holder = g.get_tensor_by_name("left_image_holder:0")
            right_image_holder = g.get_tensor_by_name("right_image_holder:0")
            label_holder = g.get_tensor_by_name("label_holder:0")
#            label_train = g.get_tensor_by_name("label_train:0")
            train_loss = tf.get_collection("train_loss")[0]
            left_train_output = tf.get_collection("left_train_output")[0]
            right_train_output = tf.get_collection("right_train_output")[0]
            
#            left_val = g.get_tensor_by_name("left_val:0")
#            right_val = g.get_tensor_by_name("right_val:0")
#            label_val = g.get_tensor_by_name("label_val:0")
#            val_loss = tf.get_collection("val_loss")[0]
            
            handle= g.get_tensor_by_name("handle:0")
    
    with tf.device(gpu_device_name):
    # create placeholders
#        image_holder = tf.placeholder(dtype=tf.float32, shape=[None, image_dims[1], image_dims[1], image_dims[-1]], name="image_holder") 
#        label_holder = tf.placeholder(dtype=tf.int64, shape=[None], name="label_holder")
#        handle = tf.placeholder(tf.string, shape=[],name="handle")
        
        # Setup tensorflow's batch generator
        train_match_dataset = tf.data.Dataset.from_tensor_slices(generator.match_train)
        train_match_dataset = train_match_dataset.shuffle(buffer_size=np.shape(generator.match_train)[0])
        train_match_dataset = train_match_dataset.repeat()
        train_match_dataset = train_match_dataset.batch(batch_size_train // 2)
        
        train_match_iterator = train_match_dataset.make_one_shot_iterator()
        iterator = tf.data.Iterator.from_string_handle(handle, train_match_dataset.output_types)
        next_element = iterator.get_next()
        
        train_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.no_match_train)
        train_non_match_dataset = train_non_match_dataset.shuffle(buffer_size=np.shape(generator.no_match_train)[0])
        train_non_match_dataset = train_non_match_dataset.repeat()
        train_non_match_dataset = train_non_match_dataset.batch(batch_size_train // 2)
        
        train_non_match_iterator = train_non_match_dataset.make_one_shot_iterator()
        
        val_match_dataset_length = np.shape(generator.match_val)[0]
        val_match_dataset = tf.data.Dataset.from_tensor_slices(generator.match_val)
        val_match_dataset = val_match_dataset.shuffle(buffer_size = val_match_dataset_length)
        val_match_dataset = val_match_dataset.repeat()
        val_match_dataset = val_match_dataset.batch(batch_size_val)
        
        val_non_match_dataset_length = np.shape(generator.no_match_val)[0]
        val_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.no_match_val[0:int(val_non_match_dataset_length/10)])
        val_non_match_dataset = val_non_match_dataset.shuffle(buffer_size = val_non_match_dataset_length)
        val_non_match_dataset = val_non_match_dataset.repeat()
        val_non_match_dataset = val_non_match_dataset.batch(batch_size_val)
        
        val_match_iterator = val_match_dataset.make_one_shot_iterator()
        val_non_match_iterator = val_non_match_dataset.make_one_shot_iterator()
        
    
#    caps1_n_maps = 32 
#    caps1_n_caps = caps1_n_maps * 42 * 42
#    caps1_n_dims = 8
#    
#    with tf.device(gpu_device_name):
#        conv1 = tf.layers.conv2d(
#            inputs = image_holder,
#            filters = 256,
#            kernel_size = [9,9], 
#            strides = [2,2],
#            padding = "valid",
#            activation = tf.nn.relu,
#            name="conv1")
#        
#        conv2 = tf.layers.conv2d(
#                inputs = conv1,
#                filters = caps1_n_maps*caps1_n_dims,
#                kernel_size = [9,9],
#                strides = [2,2],
#                padding = "valid",
#                activation = tf.nn.relu,
#                name = "conv2")
#        
#        caps1_raw = tf.reshape(conv2,[-1,caps1_n_caps,caps1_n_dims],name="caps1_raw")
#        caps1_output = squash(caps1_raw, name="caps1_output")
#        
#        caps2_n_caps = 10
#        caps2_n_dims = 16
#        
#        W_init = tf.random_normal(shape=[1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims],
#                                  stddev=0.1, dtype=tf.float32, name = "W_init")
#        W = tf.Variable(W_init,trainable=True, name="W") # transformation matrices, will be trained with backpropagation
#        W_tiled = tf.tile(W, [batch_size_train,1,1,1,1], name="W_tiled")
#        
#        # add additional dimensions to output of caps1 to conform with the dimensions of W_tiled
#        caps1_output_expanded = tf.expand_dims(caps1_output, axis=-1, name="caps1_output_expanded")
#        caps1_output_tile = tf.expand_dims(caps1_output_expanded, axis=2, name="caps1_output_tile")
#        caps1_output_tiled = tf.tile(caps1_output_tile, [1,1,caps2_n_caps,1,1], name="caps1_output_tiled")
#        
#        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")
#        
#        nbr_of_routing_iterations = 2
#        counter = 1
#        b_0 = tf.zeros([batch_size_train, caps1_n_caps, caps2_n_caps, 1, 1],dtype=tf.float32)
#    
#        def condition(b_coeff, counter):
#            return tf.less(counter, 100)
#    
#        def loop_body(b_coeff, counter):
#            routing_weights = tf.nn.softmax(b_coeff, axis=2, name="routing_weights")
#            weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
#            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True, name="weighted_sum")
#            v = squash(weighted_sum, axis=-2)
#            v_tiled = tf.tile(v, [1, caps1_n_caps, 1, 1, 1], name="v_tiled")
#            agreement = tf.matmul(caps2_predicted, v_tiled, transpose_a=True, name="agreement")
#            new_b_coeff = tf.add(b_coeff, agreement)
#            return [new_b_coeff,tf.add(1,counter)]
#        
#        b_final = tf.while_loop(condition, loop_body, [b_0, counter], maximum_iterations=nbr_of_routing_iterations)
#            
#        routing_weights = tf.nn.softmax(b_final[0], axis=2)
#        weighted_predictions = tf.multiply(routing_weights, caps2_predicted)
#        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True)
#        caps2_output = squash(weighted_sum, axis=-2)
            
#        y_prob = safe_norm(caps2_output, axis=-2)
#        y_prob_argmax = tf.argmax(y_prob, axis=2)
#        y_pred = tf.squeeze(y_prob_argmax, axis=[1,2])
#        
#        gt = tf.one_hot(label_holder, depth=caps2_n_caps)
#        loss = margin_loss(caps2_output, gt)
        
#        correct = tf.equal(label_holder, y_pred, name="correct")
#        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        
#        optimizer = tf.train.AdamOptimizer()
#        train_op = optimizer.minimize(train_loss, name="train_op")
        
#        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    
#    total_parameters = 0
#    for variable in tf.trainable_variables():
#        # shape is an array of tf.Dimension
#        shape = variable.get_shape()
#        print(shape)
#        print(len(shape))
#        variable_parameters = 1
#        for dim in shape:
#            print(dim)
#            variable_parameters *= dim.value
#        print(variable_parameters)
#        total_parameters += variable_parameters
#    print(total_parameters)
    
    with tf.Session(config=config) as sess:
        with tf.device(gpu_device_name):
#            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            if is_model_new:
#                with tf.device(gpu_device_name):
#                train_op = cu.momentum_training(train_loss, learning_rate, momentum)
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.minimize(train_loss, name="train_op")
                sess.run(tf.global_variables_initializer()) # initialize all trainable parameters
#                init.run()
                tf.add_to_collection("train_op",train_op)
            else:
#                with tf.device(gpu_device_name):
                saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                train_op = tf.get_collection("train_op")[0]
        
#        with tf.device(gpu_device_name):
#            if not os.path.exists(output_dir + "checkpoint"):
#                print("No previous model exists, creating a new one.")
#                init.run()
#            else:
#                saver.restore(sess, output_dir)

            train_match_handle = sess.run(train_match_iterator.string_handle())
            train_non_match_handle = sess.run(train_non_match_iterator.string_handle())
            val_match_handle = sess.run(val_match_iterator.string_handle())
            val_non_match_handle = sess.run(val_non_match_iterator.string_handle())
            
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

            summary_train_loss = tf.summary.scalar('training_loss', train_loss)
#            summary_op = tf.summary.scalar('training_loss', train_loss)
            
            summary_op = tf.summary.merge([summary_train_loss, filter1, hist_conv1, hist_bias1, hist_conv2, hist_bias2])
            train_writer = tf.summary.FileWriter(output_dir + "train_summary", graph=tf.get_default_graph())
            
            # training loop 
            precision_over_time = []
            val_loss_over_time = []
            start_time_train = time.time()
            for i in range(1, train_itr + 1):
#                image_batch, gt_batch = sess.run(next_element,feed_dict={handle:train_match_handle})
                
                train_batch_matching = sess.run(next_element,feed_dict={handle:train_match_handle})
                gt_matching = np.ones((np.shape(train_batch_matching)[0]),dtype=np.int32)
                train_batch_non_matching = sess.run(next_element,feed_dict={handle:train_non_match_handle})
                gt_non_matching = np.zeros((np.shape(train_batch_non_matching)[0]),dtype=np.int32)
                
                train_batch = np.append(train_batch_matching,train_batch_non_matching,axis=0)
                gt_train_batch = np.append(gt_matching,gt_non_matching,axis=0)
                permutation = np.random.permutation(batch_size_train)
                train_batch = np.take(train_batch,permutation,axis=0)
                gt_train_batch = np.take(gt_train_batch,permutation,axis=0)
                
                # Randomize rotation of batch              
                rnd_rotation = np.random.randint(0,generator.rotation_res)
                b_l_train,b_r_train = generator.get_pairs(generator.train_data[rnd_rotation],train_batch)
                
                _, train_loss_value, summary = sess.run([train_op, train_loss, summary_op], feed_dict={left_image_holder:b_l_train, right_image_holder:b_r_train, label_holder:gt_train_batch})
                
                
                if use_time:
                    elapsed_time = (time.time() - start_time_train)/60.0 # elapsed time in minutes since start of training 
                    if elapsed_time >= int(argv[-1]):
                        if meta_file_exists:
                            save_path = tf.train.Saver().save(sess,output_dir + "model",global_step=i+current_itr,write_meta_graph=False)
                        else:
                            save_path = tf.train.Saver().save(sess,output_dir + "model",global_step=i+current_itr)
                        print("Trained model after {} iterations and {} minutes saved in path: {}".format(i,elapsed_time,save_path))
                        break
                
                if i % save_itr == 0 or i == train_itr:
                    if meta_file_exists:
                        save_path = tf.train.Saver().save(sess,output_dir + "model",global_step=i+current_itr,write_meta_graph=False)
                    else:
                        save_path = tf.train.Saver().save(sess,output_dir + "model",global_step=i+current_itr)
                        meta_file_exists = True
                    print("Trained model after {} iterations saved in path: {}".format(i,save_path))
                
                
                # Use validation data set to tune hyperparameters (Classification threshold)
                if i % val_itr == 0:
                    current_val_loss = 0
    #                b_sim_val_matching = np.repeat(np.ones((batch_size_val*int(val_match_dataset_length/batch_size_val),1)),generator.rotation_res,axis=0)
    #                b_sim_val_non_matching = np.repeat(np.zeros((batch_size_val*int(val_match_dataset_length/batch_size_val),1)),generator.rotation_res,axis=0)
    #                b_sim_full = np.append(b_sim_val_matching,b_sim_val_non_matching,axis=0)
                    b_sim_val_matching = np.repeat(np.ones(batch_size_val*nbr_val_itr),generator.rotation_res,axis=0)
                    b_sim_val_non_matching = np.repeat(np.zeros((batch_size_val*nbr_val_itr)),generator.rotation_res,axis=0)
                    b_sim_full = np.append(b_sim_val_matching,b_sim_val_non_matching,axis=0)
                    for j in range(nbr_val_itr):
                        val_batch_matching = sess.run(next_element,feed_dict={handle:val_match_handle})
                        class_id_batch = generator.same_class(val_batch_matching)
                        for k in range(generator.rotation_res):
                            b_l_val,b_r_val = generator.get_pairs(generator.val_data[k],val_batch_matching) 
                            left_o,right_o,val_loss_value = sess.run([left_train_output,right_train_output, train_loss],feed_dict = {left_image_holder:b_l_val, right_image_holder:b_r_val, label_holder:np.ones(batch_size_val)})
                            current_val_loss += val_loss_value
                            if j == 0 and k == 0:
                                left_full = left_o
                                right_full = right_o
                                class_id = class_id_batch
                            else:
                                left_full = np.vstack((left_full,left_o))
                                right_full = np.vstack((right_full,right_o))
                                class_id = np.vstack((class_id, class_id_batch))
                        
                    for j in range(nbr_val_itr):
                        val_batch_non_matching = sess.run(next_element,feed_dict={handle:val_non_match_handle})
                        for k in range(generator.rotation_res):
                            b_l_val,b_r_val = generator.get_pairs(generator.val_data[0],val_batch_non_matching) 
                            left_o,right_o,val_loss_value = sess.run([left_train_output,right_train_output, train_loss],feed_dict = {left_image_holder:b_l_val, right_image_holder:b_r_val, label_holder:np.zeros(batch_size_val)})
                            left_full = np.vstack((left_full,left_o))
                            right_full = np.vstack((right_full,right_o)) 
                            class_id_batch = generator.same_class(val_batch_non_matching)
                            class_id = np.vstack((class_id,class_id_batch))
                            current_val_loss += val_loss_value
                            
                    val_loss_over_time.append(current_val_loss*batch_size_val/np.shape(b_sim_full)[0])
                    precision, false_pos, false_neg, recall, fnr, fpr, inter_class_errors = ce.get_test_diagnostics(left_full,right_full, b_sim_full,threshold, class_id)
                
                    if false_pos > false_neg:   # Can use inter_class_errors to tune the threshold further
                        threshold -= thresh_step
                    else:
                        threshold += thresh_step
                    precision_over_time.append(precision)
                    
                train_writer.add_summary(summary, i)
                
                
        # Plot precision over time
        time_points = list(range(len(precision_over_time)))
        plt.plot(time_points, precision_over_time)
        plt.title("Precision over time")
        plt.xlabel("iteration")
        plt.ylabel("precision")
        plt.show()

        print("Suggested threshold from hyperparameter tuning: %f" % threshold)
        if len(precision_over_time) > 0:
            print("Final (last computed) precision: %f" % precision_over_time[-1])
        
        # Plot validation loss over time
        plt.figure()
        plt.plot(list(range(val_itr,val_itr*len(val_loss_over_time)+1,val_itr)),val_loss_over_time)
        plt.title("Validation loss (contrastive loss) over time")
        plt.xlabel("iteration")
        plt.ylabel("validation loss")
        plt.show()
                
if __name__ == "__main__":
     main(sys.argv[1:])