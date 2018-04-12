#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  12 11:55:13 2018

@author: Tuong Lam & Simon Nilsson
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
    
    image_dims = np.shape(generator.train_data)
    
    # parameters for training
    batch_size_train = 10    # OBS! Has to be multiple of 2
    train_itr = 500000000
    
    learning_rate = 0.000001
    momentum = 0.99
    
    # Hyper parameters
    routing_iterations = 3
    digit_caps_classes = 100
    digit_caps_dims = 8
    
    caps1_n_maps = 16 
    caps1_n_dims = 8
    
    # Paramters for validation set
    batch_size_val = 10
    val_itr = 200
    threshold = 0.5
    thresh_step = 0.001
    nbr_val_itr = 10
    
    save_itr = 3000 # frequency in which the model is saved

    tf.reset_default_graph()
    
    if not os.path.exists(output_dir + "checkpoint"):
        print("No previous model exists, creating a new one.")
        is_model_new = True
        meta_file_exists = False
        current_itr = 0 # current training iteration
        
        with tf.device(gpu_device_name):
            # create placeholders
            anchor_holder, positive_holder, negative_holder, label_holder, handle = cu.placeholder_inputs(image_dims)
              
            # Create CapsNet graph
            anchor_output = cm.capsule_net(anchor_holder, routing_iterations, digit_caps_classes, digit_caps_dims, 
                                               caps1_n_maps, caps1_n_dims, batch_size_train, name="anchor_output")
            positive_output = cm.capsule_net(positive_holder, routing_iterations, digit_caps_classes, digit_caps_dims,
                                                caps1_n_maps, caps1_n_dims, batch_size_train, name="positive_output")
            negative_output = cm.capsule_net(negative_holder, routing_iterations, digit_caps_classes, digit_caps_dims,
                                                caps1_n_maps, caps1_n_dims, batch_size_train, name="negative_output")
            
            
#            # Create Reconstruction graph
#            shape = anchor_output.get_shape().as_list()[1:]
#            shape.insert(0, None)
#            reconstruct_holder_left = tf.placeholder(dtype=tf.float32, shape=shape, name="reconstruct_holder_left") 
#            reconstruct_holder_right = tf.placeholder(dtype=tf.float32, shape=shape, name="reconstruct_holder_right") 
#            reconstruct_left = cm.reconstruction_net(reconstruct_holder_left, image_dims[2:4])
#            reconstruct_right = cm.reconstruction_net(reconstruct_holder_right, image_dims[2:4])
            
            # Placeholders reconstruct images
#            shape = reconstruct_left.get_shape().as_list()[1:]
#            shape.insert(0, None)
#            image_left = tf.placeholder(dtype=tf.float32, shape = shape, name ="image_left")
#            image_right= tf.placeholder(dtype=tf.float32, shape = shape, name ="image_right")
            
            '''Triplet loss'''
            margin = tf.constant(4.0)
            train_loss = cu.triplet_caps_loss(anchor_output, positive_output, negative_output, margin)
            
            # Create loss function
            '''Contrastive loss'''
#            margin = tf.constant(3.0)
#            train_loss = cu.contrastive_caps_loss(anchor_output, positive_output, label_holder, margin)
            '''Scaled pair loss'''
#            train_loss = cu.scaled_pair_loss(anchor_output, positive_output, label_holder)
            
            # Add reconstruction loss
#            alpha = tf.constant(0.2)      # Scaling parameter of reconstructions contribution to the total loss
#            train_loss += cu.reconstruction_loss(anchor_holder, positive_holder, reconstruct_left, reconstruct_right, alpha)
#            train_loss += cu.reconstruction_loss(anchor_holder, positive_holder, image_left, image_right, alpha)
            
            # Add regularization terms to loss function
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            for i in range(len(reg_losses)):
                train_loss += reg_losses[i]
            
#            train_loss = cu.scaled_pair_loss(anchor_output, positive_output, label_holder)
            
            tf.add_to_collection("train_loss",train_loss)
            tf.add_to_collection("anchor_output",anchor_output)
            tf.add_to_collection("positive_output",positive_output)
            tf.add_to_collection("negative_output",negative_output)
#            tf.add_to_collection("reconstruct_left",reconstruct_left)
#            tf.add_to_collection("reconstruct_right",reconstruct_right)
            
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
            anchor_holder = g.get_tensor_by_name("anchor_holder:0")
            positive_holder = g.get_tensor_by_name("positive_holder:0")
            negative_holder = g.get_tensor_by_name("negative_holder:0")
            label_holder = g.get_tensor_by_name("label_holder:0")
            train_loss = tf.get_collection("train_loss")[0]
            anchor_output = tf.get_collection("anchor_output")[0]
            positive_output = tf.get_collection("positive_output")[0]
            negative_output = tf.get_collection("negative_output")[0]
            
#            reconstruct_left = tf.get_collection("reconstruct_left")[0]
#            reconstruct_right = tf.get_collection("reconstruct_right")[0]
#            reconstruct_holder_left = g.get_tensor_by_name("reconstruct_holder_left:0")
#            reconstruct_holder_right = g.get_tensor_by_name("reconstruct_holder_right:0")
#            image_left = g.get_tensor_by_name("image_left:0")
#            image_right= g.get_tensor_by_name("image_right:0")
            
            handle= g.get_tensor_by_name("handle:0")
    
    with tf.device(gpu_device_name):
        # Setup tensorflow's batch generator
        train_dataset = tf.data.Dataset.from_tensor_slices(generator.match_train)
        train_dataset = train_dataset.shuffle(buffer_size=np.shape(generator.match_train)[0])
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(batch_size_train // 2)
        
        train_iterator = train_dataset.make_one_shot_iterator()
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types)
        next_element = iterator.get_next()
        
#        train_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.no_match_train)
#        train_non_match_dataset = train_non_match_dataset.shuffle(buffer_size=np.shape(generator.no_match_train)[0])
#        train_non_match_dataset = train_non_match_dataset.repeat()
#        train_non_match_dataset = train_non_match_dataset.batch(batch_size_train // 2)
        
#        train_non_match_iterator = train_non_match_dataset.make_one_shot_iterator()
        
        val_dataset_length = np.shape(generator.match_val)[0]
        val_dataset = tf.data.Dataset.from_tensor_slices(generator.match_val)
        val_dataset = val_dataset.shuffle(buffer_size = val_dataset_length)
        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.batch(batch_size_val)
        
        val_iterator = val_dataset.make_one_shot_iterator()
        
#        val_non_match_dataset_length = np.shape(generator.no_match_val)[0]
#        val_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.no_match_val[0:int(val_non_match_dataset_length/10)])
#        val_non_match_dataset = val_non_match_dataset.shuffle(buffer_size = val_non_match_dataset_length)
#        val_non_match_dataset = val_non_match_dataset.repeat()
#        val_non_match_dataset = val_non_match_dataset.batch(batch_size_val)
        
#        val_non_match_iterator = val_non_match_dataset.make_one_shot_iterator()
        
        saver = tf.train.Saver()
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        with tf.device(gpu_device_name):
#            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            if is_model_new:
#                train_op = cu.momentum_training(train_loss, learning_rate, momentum)
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.minimize(train_loss, name="train_op")
                sess.run(tf.global_variables_initializer()) # initialize all trainable parameters
                tf.add_to_collection("train_op",train_op)
            else:
                saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                train_op = tf.get_collection("train_op")[0]
        
            train_handle = sess.run(train_iterator.string_handle())
#            train_non_match_handle = sess.run(train_non_match_iterator.string_handle())
            val_handle = sess.run(val_iterator.string_handle())
#            val_non_match_handle = sess.run(val_non_match_iterator.string_handle())
            
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
                train_batch = sess.run(next_element,feed_dict={handle:train_handle})
                gt_positive = np.ones((np.shape(train_batch)[0]),dtype=np.int32)
#                train_batch_non_matching = sess.run(next_element,feed_dict={handle:train_non_match_handle})
                gt_negative = np.zeros((np.shape(train_batch)[0]),dtype=np.int32)
                
#                train_batch = np.append(train_batch,train_batch_non_matching,axis=0)
                gt_train_batch = np.append(gt_positive,gt_negative,axis=0)
                permutation = np.random.permutation(batch_size_train)
                train_batch = np.take(train_batch,permutation,axis=0)
                gt_train_batch = np.take(gt_train_batch,permutation,axis=0)
                
                # Randomize rotation of batch              
                rnd_rotation = np.random.randint(0,generator.rotation_res)
                b_l_train,b_r_train = generator.get_pairs(generator.train_data[rnd_rotation],train_batch)
                
#                _, train_loss_value, summary = sess.run([train_op, train_loss, summary_op], 
#                                                        feed_dict={anchor_holder:b_l_train, 
#                                                                   positive_holder:b_r_train, 
#                                                                   label_holder:gt_train_batch})
                
#                left_o, right_o = sess.run([anchor_output, positive_output], 
#                                                        feed_dict={anchor_holder:b_l_train, 
#                                                                   positive_holder:b_r_train})
#                recon_left, recon_right = sess.run([reconstruct_left, reconstruct_right], 
#                                                   feed_dict={reconstruct_holder_left:left_o, 
#                                                              reconstruct_holder_right:right_o})
#                _, train_loss_value, summary = sess.run([train_op, train_loss, summary_op], 
#                                               feed_dict={anchor_holder:b_l_train,
#                                                          positive_holder:b_r_train,
#                                                          image_left:recon_left,
#                                                          image_right:recon_right,
#                                                          label_holder:gt_train_batch,
#                                                          reconstruct_holder_left:left_o,
#                                                          reconstruct_holder_right:right_o})            
    
                
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
#                if i % val_itr == 0:
#                    current_val_loss = 0
#    #                b_sim_val_matching = np.repeat(np.ones((batch_size_val*int(val_dataset_length/batch_size_val),1)),generator.rotation_res,axis=0)
#    #                b_sim_val_non_matching = np.repeat(np.zeros((batch_size_val*int(val_dataset_length/batch_size_val),1)),generator.rotation_res,axis=0)
#    #                b_sim_full = np.append(b_sim_val_matching,b_sim_val_non_matching,axis=0)
#                    b_sim_val_matching = np.repeat(np.ones(batch_size_val*nbr_val_itr),generator.rotation_res,axis=0)
#                    b_sim_val_non_matching = np.repeat(np.zeros((batch_size_val*nbr_val_itr)),generator.rotation_res,axis=0)
#                    b_sim_full = np.append(b_sim_val_matching,b_sim_val_non_matching,axis=0)
#                    for j in range(nbr_val_itr):
#                        val_batch_matching = sess.run(next_element,feed_dict={handle:val_match_handle})
#                        class_id_batch = generator.same_class(val_batch_matching)
#                        for k in range(generator.rotation_res):
#                            b_l_val,b_r_val = generator.get_pairs(generator.val_data[k],val_batch_matching) 
##                            left_o,right_o,val_loss_value = sess.run([anchor_output,positive_output, train_loss],
##                                                                     feed_dict = {anchor_holder:b_l_val, positive_holder:b_r_val, label_holder:np.ones(batch_size_val)})
#                            left_o,right_o = sess.run([anchor_output,positive_output],
#                                                                     feed_dict = {anchor_holder:b_l_val, positive_holder:b_r_val})
##                            current_val_loss += val_loss_value
#                            if j == 0 and k == 0:
#                                left_full = left_o
#                                right_full = right_o
#                                class_id = class_id_batch
#                            else:
#                                left_full = np.vstack((left_full,left_o))
#                                right_full = np.vstack((right_full,right_o))
#                                class_id = np.vstack((class_id, class_id_batch))
#                        
#                    for j in range(nbr_val_itr):
#                        val_batch_non_matching = sess.run(next_element,feed_dict={handle:val_non_match_handle})
#                        for k in range(generator.rotation_res):
#                            b_l_val,b_r_val = generator.get_pairs(generator.val_data[0],val_batch_non_matching) 
##                            left_o,right_o,val_loss_value = sess.run([anchor_output,positive_output, train_loss],
##                                                                     feed_dict = {anchor_holder:b_l_val, positive_holder:b_r_val, label_holder:np.zeros(batch_size_val)})
#                            left_o,right_o = sess.run([anchor_output,positive_output],
#                                                                     feed_dict = {anchor_holder:b_l_val, positive_holder:b_r_val})
#                            left_full = np.vstack((left_full,left_o))
#                            right_full = np.vstack((right_full,right_o)) 
#                            class_id_batch = generator.same_class(val_batch_non_matching)
#                            class_id = np.vstack((class_id,class_id_batch))
##                            current_val_loss += val_loss_value
#                            
#                    val_loss_over_time.append(current_val_loss*batch_size_val/np.shape(b_sim_full)[0])
#                    precision, false_pos, false_neg, recall, fnr, fpr, inter_class_errors = ce.get_test_diagnostics(left_full,right_full, b_sim_full,threshold, class_id)
#                
#                    if false_pos > false_neg:   # Can use inter_class_errors to tune the threshold further
#                        threshold -= thresh_step
#                    else:
#                        threshold += thresh_step
#                    precision_over_time.append(precision)
                    
                train_writer.add_summary(summary, i)
            
#            reconstruct_images_left, reconstruct_images_right = sess.run([reconstruct_left, reconstruct_left],
#                                                                         feed_dict={reconstruct_holder_left:left_o,
#                                                                                    reconstruct_holder_right:right_o})
#            reconstruct_images_left = np.reshape(reconstruct_images_left, (-1,192,192))
#            reconstruct_images_right = np.reshape(reconstruct_images_right, (-1,192,192))
#            for k in range(10):
#                plt.imshow(reconstruct_images_left[k],cmap='Greys_r')
#                plt.show()
#                plt.imshow(b_l_train[k,:,:,0],cmap='Greys_r')
#                plt.show()
                
                
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