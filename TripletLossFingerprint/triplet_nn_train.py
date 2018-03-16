# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:44:22 2018
@author: Tuong Lam & Simon Nilsson
"""

from data_generator import data_generator

import numpy as np
import tensorflow as tf
import os 
import matplotlib.pyplot as plt
import pickle
import re
import sys
import time
import scipy.linalg as sl

import utilities as util
import triplet_nn_eval as tre
import triplet_nn_model as sm


def main(argv):
    """ This method is used to train a siamese network for fingerprint datasets.
    
    The model is defined in the file siamese_nn_model.py.
    When training is completed the model is saved in the file /tmp/siamese_finger_model/.
    If a model exists it will be used for further training, otherwise a new
    one is created. It is also possible to evaluate the model directly after training.
    
    """
    
    model_name = argv[0]
    gpu_device_name = argv[2]
    if len(argv) == 4:
        use_time = True
    else:
        use_time = False
        
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir =  argv[1] + model_name + "/" # directory where the model will be saved

    # Load fingerprint data and create a data_generator instance if one 
    # does not exist, otherwise load existing data_generator
    if not os.path.exists(dir_path + "/generator_data.pk1"):
        with open('generator_data.pk1', 'wb') as output:
            # Load fingerprint labels and data from file with names
            finger_id = np.load(dir_path + "/finger_id.npy")
            person_id = np.load(dir_path + "/person_id.npy")
            finger_data = np.load(dir_path + "/fingerprints.npy")
            translation = np.load(dir_path + "/translation.npy")
            rotation = np.load(dir_path + "/rotation.npy")
            nbr_of_images = np.shape(finger_data)[0] # number of images to use from the original data set
#            nbr_of_images = 5000
            finger_data = util.reshape_grayscale_data(finger_data)
            rotation_res = 1
            
            generator = data_generator(finger_data, finger_id, person_id, translation, rotation, nbr_of_images, rotation_res) # initialize data generator
            
            finger_id_gt_vt = np.load(dir_path + "/finger_id_gt_vt.npy")
            person_id_gt_vt = np.load(dir_path + "/person_id_gt_vt.npy")
            finger_data_gt_vt = np.load(dir_path + "/fingerprints_gt_vt.npy")
            translation_gt_vt = np.load(dir_path + "/translation_gt_vt.npy")
            rotation_gt_vt = np.load(dir_path + "/rotation_gt_vt.npy")
            
            finger_data_gt_vt = util.reshape_grayscale_data(finger_data_gt_vt)
            nbr_of_images_gt_vt = np.shape(finger_data_gt_vt)[0]
#            nbr_of_images_gt_vt = 5000            
            
            generator.add_new_data(finger_data_gt_vt, finger_id_gt_vt, person_id_gt_vt, translation_gt_vt, rotation_gt_vt, nbr_of_images_gt_vt)
            pickle.dump(generator, output, pickle.HIGHEST_PROTOCOL)
    else:
        # Load generator
        with open('generator_data.pk1', 'rb') as input:
            generator = pickle.load(input)
             
    # parameters for training
    batch_size_train = 300
    train_itr = 1000000000000
    learning_rate = 0.00001
    momentum = 0.99
    
    
    # Paramerters for increasing difficulty
#    lvl_2 = 100     # Set number of iterations at when to increase difficulty to level 2
#    lvl_3 = 200     # Set number of iterations at when to increase difficulty to level 3
    harder_itr = 300
    batch_size_increase_diff = 900
   
    # parameters for validation
    batch_size_val = 300
    val_itr = 1000 # frequency in which to use validation data for computations
    
    # parameters for evaluation
    batch_size_test = 200
    threshold = 0.23    
    thresh_step = 0.01
        
    dims = np.shape(generator.train_data[0])
    batch_sizes = [batch_size_train,batch_size_val,batch_size_test]
    image_dims = [dims[1],dims[2],dims[3]]
    
    save_itr = 25000 # frequency in which the model is saved
    
    tf.reset_default_graph()
    
    # if models exists use the latest existing one otherwise create a new one
    if not os.path.exists(output_dir + "checkpoint"):
        print("No previous model exists, creating a new one.")
        is_model_new = True
        meta_file_exists = False
        current_itr = 0 # current training iteration
        
        with tf.device(gpu_device_name):
             # create placeholders
            anchor_train,pos_train,neg_train,anchor_val,pos_val,neg_val,left_test,right_test = sm.placeholder_inputs(image_dims,batch_sizes)
            handle = tf.placeholder(tf.string, shape=[],name="handle")
                
            anchor_train_output = sm.inference(anchor_train)            
            pos_train_output = sm.inference(pos_train)
            neg_train_output = sm.inference(neg_train)
            anchor_val_output = sm.inference(anchor_val)
            pos_val_output = sm.inference(pos_val)
            neg_val_output = sm.inference(neg_val)
            
            left_test_output = sm.inference(left_test)
            right_test_output = sm.inference(right_test)
            
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            margin = tf.constant(4.0) # margin for contrastive loss
            train_loss = sm.triplet_loss(anchor_train_output,pos_train_output,neg_train_output,margin)
            # add regularization terms to triplet loss function
            for i in range(len(reg_losses)):
                train_loss += reg_losses[i]
            
            val_loss = sm.triplet_loss(anchor_val_output,pos_val_output,neg_val_output,margin)
            
            tf.add_to_collection("train_loss",train_loss)
            tf.add_to_collection("val_loss",val_loss)
            tf.add_to_collection("anchor_val_output",anchor_val_output)
            tf.add_to_collection("pos_val_output",pos_val_output)
            tf.add_to_collection("neg_val_output",neg_val_output)
            tf.add_to_collection("left_test_output",left_test_output)
            tf.add_to_collection("right_test_output",right_test_output)
            
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
            anchor_train = g.get_tensor_by_name("anchor_train:0")
            pos_train = g.get_tensor_by_name("positive_train:0")
            neg_train = g.get_tensor_by_name("negative_train:0")
            train_loss = tf.get_collection("train_loss")[0]
            anchor_val_output = tf.get_collection("anchor_val_output")[0]
            pos_val_output = tf.get_collection("pos_val_output")[0]
            neg_val_output = tf.get_collection("neg_val_output")[0]
            
            anchor_val = g.get_tensor_by_name("anchor_val:0")
            pos_val = g.get_tensor_by_name("positive_val:0")
            neg_val = g.get_tensor_by_name("negative_val:0")
            val_loss = tf.get_collection("val_loss")[0]
            
            left_test = g.get_tensor_by_name("left_test:0")
            right_test = g.get_tensor_by_name("right_test:0")
            left_test_output = tf.get_collection("left_test_output")[0]
            right_test_output = tf.get_collection("right_test_output")[0]
            
            handle = g.get_tensor_by_name("handle:0")
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
#    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
        if is_model_new:
            with tf.device(gpu_device_name):
                train_op = sm.training(train_loss, learning_rate, momentum)
                sess.run(tf.global_variables_initializer()) # initialize all trainable parameters
                tf.add_to_collection("train_op",train_op)
        else:
            with tf.device(gpu_device_name):
                saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                train_op = tf.get_collection("train_op")[0]
            
#            for i in sess.graph.get_operations():
#                print(i.values())
#            global_vars = tf.global_variables()
#            for i in range(len(global_vars)):
#                print(global_vars[i])
                
        with tf.device(gpu_device_name): 
            graph = tf.get_default_graph()
            
            # Summary setup
            conv1_filters = graph.get_tensor_by_name("conv_layer_1/kernel:0")
            nbr_of_filters_conv1 = sess.run(tf.shape(conv1_filters)[-1])
    
            conv2_filters = graph.get_tensor_by_name("conv_layer_2/kernel:0")
            hist_conv1 = tf.summary.histogram("hist_conv1", conv1_filters)
            hist_conv2 = tf.summary.histogram("hist_conv2", conv2_filters)
            conv1_filters = tf.transpose(conv1_filters, perm = [3,0,1,2])
            filter1 = tf.summary.image('Filter_1', conv1_filters, max_outputs=nbr_of_filters_conv1)
            conv1_bias = graph.get_tensor_by_name("conv_layer_1/bias:0")
            hist_bias1 = tf.summary.histogram("hist_bias1", conv1_bias)
            conv2_bias = graph.get_tensor_by_name("conv_layer_2/bias:0")
            hist_bias2 = tf.summary.histogram("hist_bias2", conv2_bias)
                
            summary_train_loss = tf.summary.scalar('training_loss', train_loss)
#            x_image = tf.summary.image('anchor_input', anchor_train)
            summary_op = tf.summary.merge([summary_train_loss, filter1, hist_conv1, hist_conv2, hist_bias1, hist_bias2])
            train_writer = tf.summary.FileWriter(output_dir + "/train_summary", graph=tf.get_default_graph())
            
            
        precision_over_time = []
        val_loss_over_time = []
        
        with tf.device(gpu_device_name):
            # Setup tensorflow's batch generator
            train_anchors_dataset = tf.data.Dataset.from_tensor_slices(generator.anchors_train)
            train_anchors_dataset = train_anchors_dataset.shuffle(buffer_size=np.shape(generator.anchors_train)[0])
            train_anchors_dataset = train_anchors_dataset.repeat()
            train_anchors_dataset = train_anchors_dataset.batch(int(batch_size_train))
            
#            train_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.no_match_train)
#            train_non_match_dataset = train_non_match_dataset.shuffle(buffer_size=np.shape(generator.no_match_train)[0])
#            train_non_match_dataset = train_non_match_dataset.repeat()
#            train_non_match_dataset = train_non_match_dataset.batch(int((batch_size_train+1)/2))
            
            val_anchors_dataset_length = np.shape(generator.anchors_val)[0]
            val_anchors_dataset = tf.data.Dataset.from_tensor_slices(generator.anchors_val)
            val_anchors_dataset = val_anchors_dataset.shuffle(buffer_size = val_anchors_dataset_length)
            val_anchors_dataset = val_anchors_dataset.repeat()
            val_anchors_dataset = val_anchors_dataset.batch(batch_size_val)
            
#            val_non_match_dataset_length = np.shape(generator.no_match_val)[0]
#            val_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.no_match_val[0:int(val_non_match_dataset_length/10)])
#            val_non_match_dataset = val_non_match_dataset.shuffle(buffer_size = val_non_match_dataset_length)
#            val_non_match_dataset = val_non_match_dataset.repeat()
#            val_non_match_dataset = val_non_match_dataset.batch(batch_size_val)
            
            train_anchors_iterator = train_anchors_dataset.make_one_shot_iterator()
            train_anchors_handle = sess.run(train_anchors_iterator.string_handle())
            
            val_anchors_iterator = val_anchors_dataset.make_one_shot_iterator()
            val_anchors_handle = sess.run(val_anchors_iterator.string_handle())
            
#            train_non_match_iterator = train_non_match_dataset.make_one_shot_iterator()
#            train_non_match_handle = sess.run(train_non_match_iterator.string_handle())
            
#            val_non_match_iterator = val_non_match_dataset.make_one_shot_iterator()
#            val_non_match_handle = sess.run(val_non_match_iterator.string_handle())
            
            iterator = tf.data.Iterator.from_string_handle(handle, train_anchors_dataset.output_types)
            next_element = iterator.get_next()
        
        print("Starting training")
        start_time_train = time.time()
        # Training loop
        for i in range(1,train_itr + 1):
            train_batch_anchors = sess.run(next_element,feed_dict={handle:train_anchors_handle})
#            b_sim_train_matching = np.ones((np.shape(train_batch_matching)[0],1),dtype=np.int32)
#            train_batch_non_matching = sess.run(next_element,feed_dict={handle:train_non_match_handle})
#            b_sim_train_non_matching = np.zeros((np.shape(train_batch_non_matching)[0],1),dtype=np.int32)
            
#            train_batch = np.append(train_batch_matching,train_batch_non_matching,axis=0)
#            b_sim_train = np.append(b_sim_train_matching,b_sim_train_non_matching,axis=0)
#            permutation = np.random.permutation(batch_size_train)
#            train_batch = np.take(train_batch,permutation,axis=0)
#            b_sim_train = np.take(b_sim_train,permutation,axis=0)
            
            ####### Increase difficulty every harder_itr iteration by offline evaluation on a subset of all triplets #######
            if i % harder_itr == 0:                   
                # Take a random subset of each anchors non matching set
#                nbr_non_matching = 30
#                for j in range(len(generator.triplets_train_original)):
#                    no_match_samples = np.random.choice(generator.triplets_train_original[j][1], nbr_non_matching)
##                    no_match_samples = no_match_samples.reshape((nbr_non_matching,1))
#                    if j == 0:
#                        negative_pairs = np.array([j*np.ones((nbr_non_matching)), no_match_samples], dtype = 'int32')
#                    else:
#                        negative_pairs = np.hstack((negative_pairs, np.array([j*np.ones((nbr_non_matching)), no_match_samples], dtype = 'int32')))
#                negative_pairs = negative_pairs.T
                
                breakpoint_match = []
                nbr_non_matching = 20
                for j in range(len(generator.anchors_train)):
                    current_image = generator.anchors_train[j]
                    no_match_samples = np.random.choice(generator.triplets_train_original[current_image][1], nbr_non_matching)
                    match = generator.triplets_train_original[current_image][0]
                    if j == 0:
                        negative = np.array(no_match_samples, dtype = 'int32')
                        positive = np.array(match, dtype = 'int32')
                        breakpoint_match.append(len(match))
                    else:
                        negative = np.hstack((negative, np.array(no_match_samples, dtype = 'int32')))
                        positive = np.hstack((positive, np.array(match, dtype = 'int32')))
                        breakpoint_match.append(len(match) + breakpoint_match[-1])
#                negative = negative.T
#                positive = positive.T
                    
                for j in range(int(negative.shape[0] / batch_size_test) + 1):
                    batch_start = j*batch_size_increase_diff
                    batch_stop = (j+1)*batch_size_increase_diff
                    if batch_stop > negative.shape[0]:
                        batch_stop = negative.shape[0]
                        if batch_stop <= batch_start:
                            break
                    b_neg = generator.get_images(generator.train_data[0], negative[batch_start:batch_stop])
                    neg_o = sess.run(right_test_output, feed_dict = {right_test:b_neg})
                    if j == 0:
                        neg_full = neg_o
                    else:
                        neg_full = np.vstack((neg_full,neg_o))
                        
                for j in range(int(len(generator.anchors_train) / batch_size_test) + 1):
                    batch_start = j*batch_size_increase_diff
                    batch_stop = (j+1)*batch_size_increase_diff
                    if batch_stop > len(generator.anchors_train):
                        batch_stop = len(generator.anchors_train)
                        if batch_stop <= batch_start:
                            break
                    b_anch = generator.get_images(generator.train_data[0], generator.anchors_train[batch_start:batch_stop])
                    anch_o = sess.run(right_test_output, feed_dict = {right_test:b_anch})
                    if j == 0:
                        anch_full = anch_o
                    else:
                        anch_full = np.vstack((anch_full,anch_o))
                
                for j in range(int(len(positive) / batch_size_test) + 1):
                    batch_start = j*batch_size_increase_diff
                    batch_stop = (j+1)*batch_size_increase_diff
                    if batch_stop > len(positive):
                        batch_stop = len(positive)
                        if batch_stop <= batch_start:
                            break
                    b_pos = generator.get_images(generator.train_data[0], positive[batch_start:batch_stop])
                    pos_o = sess.run(right_test_output, feed_dict = {right_test:b_pos})
                    if j == 0:
                        pos_full = pos_o
                    else:
                        pos_full = np.vstack((pos_full,pos_o))
                        
                distance_neg = []
                for j in range(len(generator.anchors_train)):
                    dist = sl.norm(neg_full[j*nbr_non_matching:(j+1)*nbr_non_matching] - anch_full[j], axis=1)
                    distance_neg.append(dist) 
                
                distance_pos = []
                for j in range(len(generator.anchors_train)):
                    if j > 0:
                        dist = sl.norm(pos_full[breakpoint_match[j-1]:breakpoint_match[j]] - anch_full[j], axis=1)
                    else:
                        dist = sl.norm(pos_full[0:breakpoint_match[j]] - anch_full[j], axis=1)
                    distance_pos.append(dist) 
                    
                hardest_neg = []
                nbr_hardest_neg = 1
                for j in range(len(distance_neg)):
                    hardest_current = np.full((nbr_hardest_neg,2), np.inf)        # Keeps track on index in first column and distance in second
                    for k in range(nbr_non_matching):
                        if distance_neg[j][k] < hardest_current[0][1]:
                            hardest_current[0] = [negative[j*nbr_non_matching+k],distance_neg[j][k]]
                            hardest_current = hardest_current[hardest_current[:,1].argsort()][::-1]  # Sort in decending order based on distance
                            
#                    for k in range(j*nbr_non_matching, (j+1)*nbr_non_matching):
#                        if distance_neg[k] < hardest_current[0][1]:
#                            hardest_current[0] = [negative[k],distance_neg[k]]
#                            hardest_current = hardest_current[hardest_current[:,1].argsort()][::-1]  # Sort in decending order based on distance
                    hardest_neg.append(hardest_current[:,0].astype('int32'))
                
                hardest_pos = []
                maximum_nbr_hardest_pos = 1
                for j in range(len(generator.anchors_train)):
                    if j > 0:
                        match_dist = distance_pos[j]
                        hardest_current = np.array([positive[breakpoint_match[j-1]:breakpoint_match[j]], match_dist]).T
                    else:
                        match_dist = distance_pos[j]
                        hardest_current = np.array([positive[0:breakpoint_match[j]], match_dist]).T
                    
                    hardest_current = hardest_current[hardest_current[:,1].argsort()][::-1]  # Sort in decending order based on distance
                    if hardest_current.shape[0] > maximum_nbr_hardest_pos:
                        hardest_pos.append(hardest_current[0:maximum_nbr_hardest_pos,0].astype('int32'))
                    else:
                        hardest_pos.append(hardest_current[:,0].astype('int32'))
                    
                generator.update_triplets(hardest_neg, hardest_pos)
                
#                # Run network on the subset and save the output
#                for j in range(int(len(negative_pairs) / batch_size_test)):
#                    b_anch, b_neg = generator.get_pairs(generator.train_data[0], negative_pairs[j*batch_size_test:(j+1)*batch_size_test])
#                
#                    anchor_o,neg_o = sess.run([left_test_output, right_test_output],feed_dict = {left_test:b_anch, right_test:b_neg})
#                    if j == 0:
#                        anchor_full = anchor_o
#                        neg_full = neg_o
#                    else:
#                        anchor_full = np.vstack((anchor_full,anchor_o))
#                        neg_full = np.vstack((neg_full,neg_o))
                
                # Calculate distance between all non matching pairs
#                distance = sl.norm(anchor_full - neg_full,axis=1)
                
                # Create the new non matching set and replace in the generator
#                hardest_all = []
#                nbr_hardest = 5
#                for j in range(int(len(distance) / nbr_non_matching)):
#                    hardest_current = np.full((nbr_hardest,2), np.inf)        # Keeps track on index in first column and distance in second
#                    for k in range(j*nbr_non_matching, (j+1)*nbr_non_matching):
#                        if distance[k] < hardest_current[0][1]:
#                            hardest_current[0] = [negative_pairs[k][1],distance[k]]
#                            hardest_current = hardest_current[hardest_current[:,1].argsort()][::-1]  # Sort in decending order based on distance
#                    hardest_all.append(hardest_current[:,0].astype('int32'))
#                generator.update_no_match(hardest_all)        
                
                
#            if i < lvl_2:
#                difficulty_lvl = 1
#            elif i < lvl_3:
#                difficulty_lvl = 2
#            else:
#                difficulty_lvl = 3   
            # Randomize rotation of batch              
            rnd_rotation = np.random.randint(0,generator.rotation_res)
#            b_anch_train,b_pos_train,b_neg_train = generator.get_triplet(generator.train_data[rnd_rotation], generator.triplets_train, train_batch_anchors, difficulty_lvl)
            b_anch_train,b_pos_train,b_neg_train = generator.get_triplet(generator.train_data[rnd_rotation], generator.triplets_train, train_batch_anchors)
            
            _,train_loss_value, summary = sess.run([train_op, train_loss, summary_op],feed_dict={anchor_train:b_anch_train, pos_train:b_pos_train, neg_train:b_neg_train})

             # Use validation data set to tune hyperparameters (Classification threshold)
            if i % val_itr == 0:
                current_val_loss = 0
#                b_sim_val_anchors = np.repeat(np.ones((batch_size_val*int(val_anchors_dataset_length/batch_size_val),1)),generator.rotation_res,axis=0)
#                b_sim_val_non_matching = np.repeat(np.zeros((batch_size_val*int(val_match_dataset_length/batch_size_val),1)),generator.rotation_res,axis=0)
#                b_sim_full = np.append(b_sim_val_matching,b_sim_val_non_matching,axis=0)
                labels = np.vstack((np.ones((batch_size_val,1)), np.zeros((batch_size_val,1))))
                for j in range(int(val_anchors_dataset_length/batch_size_val)):
                    val_batch_anchors = sess.run(next_element,feed_dict={handle:val_anchors_handle})
#                    class_id_batch = generator.same_class(val_batch_anchors)
                    for k in range(generator.rotation_res):
#                        b_l_val,b_r_val = generator.get_pairs(generator.val_data[k],val_batch_anchors) 
#                        difficulty_lvl = np.random.randint(1,4)
#                        b_anch_val,b_pos_val,b_neg_val= generator.get_triplet(generator.val_data[k], generator.triplets_val, val_batch_anchors, difficulty_lvl)
                        b_anch_val,b_pos_val,b_neg_val= generator.get_triplet(generator.val_data[k], generator.triplets_val, val_batch_anchors)
                        anchor_o,pos_o,neg_o,val_loss_value = sess.run([anchor_val_output,pos_val_output, neg_val_output,val_loss],feed_dict = {anchor_val:b_anch_val, pos_val:b_pos_val, neg_val:b_neg_val})
                        current_val_loss += val_loss_value
                        if j == 0 and k == 0:
                            anchor_full = np.vstack((anchor_o,anchor_o))
                            candidates_full = np.vstack((pos_o,neg_o))
                            labels_full = labels
#                            class_id = class_id_batch
                        else:
                            anchor_full = np.vstack((anchor_full,anchor_o,anchor_o))
                            candidates_full = np.vstack((candidates_full,pos_o,neg_o))
                            labels_full = np.vstack((labels_full,labels))
#                            class_id = np.vstack((class_id, class_id_batch))
                    
#                for j in range(int(val_match_dataset_length/batch_size_val)):
#                    val_batch_non_matching = sess.run(next_element,feed_dict={handle:val_non_match_handle})
#                    for k in range(generator.rotation_res):
#                        b_l_val,b_r_val = generator.get_pairs(generator.val_data[0],val_batch_non_matching) 
#                        anchor_o,pos_o,val_loss_value  = sess.run([anchor_val_output,pos_val_output,val_loss],feed_dict = {anchor_val:b_l_val, pos_val:b_r_val, label_val:np.zeros((batch_size_val,1))})
#                        anchor_full = np.vstack((anchor_full,anchor_o))
#                        pos_full = np.vstack((pos_full,pos_o)) 
#                        class_id_batch = generator.same_class(val_batch_non_matching)
#                        class_id = np.vstack((class_id,class_id_batch))
#                        current_val_loss += val_loss_value
                        
                val_loss_over_time.append(current_val_loss*batch_size_val/np.shape(labels_full)[0])
                precision, false_pos, false_neg, recall, fnr, fpr = tre.get_test_diagnostics(anchor_full,candidates_full, labels_full,threshold)
            
                if false_pos > false_neg:   # Can use inter_class_errors to tune the threshold further
                    threshold -= thresh_step
                else:
                    threshold += thresh_step
                precision_over_time.append(precision)

            train_writer.add_summary(summary, i)
            
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