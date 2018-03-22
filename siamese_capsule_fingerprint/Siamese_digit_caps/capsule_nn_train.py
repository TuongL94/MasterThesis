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
    
#    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = argv[1]# + "../data_manager/"
    output_dir = argv[1] + "models/" + argv[0] # directory where the model is saved
    gpu_device_name = argv[2] # gpu device to use
    
    if len(argv) == 4:
        use_time = True
    else:
        use_time = False
    
    # Load fingerprint data and create a data_generator instance if one 
    # does not exist, otherwise load existing data_generator
    if not os.path.exists(data_path + "/generator_data.pk1"):
        with open("generator_data.pk1", "wb") as output:
            # Load fingerprint label_holders and data from file with names
            finger_id = np.load(data_path + "finger_id.npy")
            person_id = np.load(data_path + "person_id.npy")
            finger_data = np.load(data_path+ "fingerprints.npy")
            translation = np.load(data_path + "translation.npy")
            rotation = np.load(data_path + "rotation.npy")
            finger_data = util.reshape_grayscale_data(finger_data)
            nbr_of_images = np.shape(finger_data)[0] # number of images to use from the original data set
            
            rotation_res = 1
            generator = data_generator(finger_data, finger_id, person_id, translation, rotation, nbr_of_images, rotation_res) # initialize data generator
            
            finger_id_gt_vt = np.load(data_path + "finger_id_gt_vt.npy")
            person_id_gt_vt = np.load(data_path+ "person_id_gt_vt.npy")
            finger_data_gt_vt = np.load(data_path + "fingerprints_gt_vt.npy")
            translation_gt_vt = np.load(data_path + "translation_gt_vt.npy")
            rotation_gt_vt = np.load(data_path + "rotation_gt_vt.npy")
            finger_data_gt_vt = util.reshape_grayscale_data(finger_data_gt_vt)
            nbr_of_images_gt_vt = np.shape(finger_data_gt_vt)[0]
            
            generator.add_new_data(finger_data_gt_vt, finger_id_gt_vt, person_id_gt_vt, translation_gt_vt, rotation_gt_vt, nbr_of_images_gt_vt)
            pickle.dump(generator, output, pickle.HIGHEST_PROTOCOL)
    else:
        # Load generator
        with open("generator_data.pk1", 'rb') as input:
            generator = pickle.load(input)
    
#    train_images, train_gt = get_classification_data(generator,10)
    image_dims = np.shape(generator.train_data)
    
    # parameters for training
    batch_size_train = 500    # OBS! Has to be multiple of 2
    train_itr = 500
    
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
    val_itr = 20
    threshold = 0.00001
    thresh_step = 0.000001
    nbr_val_itr = 1
    
    save_itr = 100 # frequency in which the model is saved

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
            
#            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#            margin = tf.constant(4.0) # margin for contrastive loss
#            train_loss = sm.triplet_loss(anchor_train_output,pos_train_output,neg_train_output,margin)
#            # add regularization terms to triplet loss function
#            for i in range(len(reg_losses)):
#                train_loss += reg_losses[i]
            
#            train_loss = cu.scaled_pair_loss(left_train_output, right_train_output, label_holder)
            
            margin = tf.constant(4.0)
            train_loss = cu.contrastive_caps_loss(left_train_output, right_train_output, label_holder, margin)
            
            tf.add_to_collection("train_loss",train_loss)
            tf.add_to_collection("left_train_output",left_train_output)
            tf.add_to_collection("right_train_output",right_train_output)
            
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
            train_loss = tf.get_collection("train_loss")[0]
            left_train_output = tf.get_collection("left_train_output")[0]
            right_train_output = tf.get_collection("right_train_output")[0]
            
            handle= g.get_tensor_by_name("handle:0")
    
    with tf.device(gpu_device_name):
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
                    if elapsed_time >= int(argv[3]):
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