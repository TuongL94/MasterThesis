# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:44:22 2018

@author: Tuong Lam & Simon Nilsson
"""

import numpy as np
import tensorflow as tf
import os 
import matplotlib.pyplot as plt
import pickle
import re
import sys
import time

# imports from self-implemented modules
import inception_nn_model as im
import inception_nn_eval as ie
import siamese_nn_utilities as su
import utilities as util
from data_generator import data_generator

def main(argv):
    """ This method is used to train an inception network for fingerprint datasets.
    
    The model is defined in the file inception_nn_model.py.
    If a model exists it will be used for further training, otherwise a new
    one is created.
    
    Input:
    argv - arguments to run this method
    argv[0] - path of the directory which the model will be saved in
    argv[1] - path of the directory where the data is located
    argv[2] - name of the GPU to use for training
    argv[3] - optional argument, if this argument is given the model will train for argv[3] minutes
              otherwise it will train for a given amount of iterations
    """
    
    output_dir = argv[0]
    data_path =  argv[1]
    gpu_device_name = argv[2]
    if len(argv) == 4:
        use_time = True
    else:
        use_time = False

    # Load fingerprint data and create a data_generator instance if one 
    # does not exist, otherwise load existing data_generator
    if not os.path.exists(data_path + "generator_data_small_rotdiff5_transdiff10_new.pk1"):
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
        with open(data_path + "generator_data_small_rotdiff5_transdiff30_new.pk1", "rb") as input:
            generator = pickle.load(input)
             
    # parameters for training
    batch_size_train = 40
    train_itr = 30000000000
    learning_rate = 0.00005
    momentum = 0.99
    
    # margin setup for contrastive loss
    margin = 0.5
    margin_factor = 1.1
    max_margin = 1.3 # maximum allowed margin value
    margin_itr = 400 # frequency in which to increase the margin in loss function 

    # parameters for validation
    batch_size_val = 200
    val_itr = 30000000000000 # frequency in which to use validation data for computations
    threshold = 0.5    
    thresh_step = 0.1
        
    dims = np.shape(generator.train_data[0])
    image_dims = [dims[1],dims[2],dims[3]]
    
    save_itr = 2000 # frequency in which the model is saved
    
    tf.reset_default_graph()
    
    # if models exists use the latest existing one otherwise create a new one
    if not os.path.exists(output_dir + "checkpoint"):
        print("No previous model exists, creating a new one.")
        is_model_new = True
        meta_file_exists = False
        current_itr = 0 # current training iteration
        
        with tf.device(gpu_device_name):
             # create placeholders
            left_train,right_train,label_train,left_val,right_val,label_val,left_test,right_test = su.create_placeholders(image_dims)
            handle = tf.placeholder(tf.string, shape=[],name="handle")
            margin_holder = tf.placeholder(tf.float32, shape=[], name="margin_holder")
                
            left_train_output = im.inference(left_train)  
#            print(util.get_nbr_of_parameters())
            right_train_output = im.inference(right_train)
            left_val_output = im.inference(left_val, training=False)
            right_val_output = im.inference(right_val, training=False)
            left_test_output = im.inference(left_test, training=False)
            right_test_output = im.inference(right_test, training=False)
            
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            train_loss = su.contrastive_loss(left_train_output,right_train_output,label_train,margin)
            
            # add regularization terms to loss function
            for i in range(len(reg_losses)):
                train_loss += reg_losses[i]
            
            val_loss = su.contrastive_loss(left_val_output,right_val_output,label_val,margin)
            
            tf.add_to_collection("train_loss",train_loss)
            tf.add_to_collection("val_loss",val_loss)
            tf.add_to_collection("left_val_output",left_val_output)
            tf.add_to_collection("right_val_output",right_val_output)
            tf.add_to_collection("left_test_output",left_test_output)
            tf.add_to_collection("right_test_output",right_test_output)
            
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
            left_train = g.get_tensor_by_name("left_train:0")
            right_train = g.get_tensor_by_name("right_train:0")
            label_train = g.get_tensor_by_name("label_train:0")
            train_loss = tf.get_collection("train_loss")[0]
            left_val_output = tf.get_collection("left_val_output")[0]
            right_val_output = tf.get_collection("right_val_output")[0]
            
            left_val = g.get_tensor_by_name("left_val:0")
            right_val = g.get_tensor_by_name("right_val:0")
            label_val = g.get_tensor_by_name("label_val:0")
            val_loss = tf.get_collection("val_loss")[0]
            
            handle= g.get_tensor_by_name("handle:0")
            margin_holder = g.get_tensor_by_name("margin_holder:0")
            
            left_train_output = tf.get_collection("left_train_output")[0]
            right_train_output = tf.get_collection("right_train_output")[0]
    
    with tf.device(gpu_device_name):
        # Setup tensorflow's batch generator
        train_match_dataset = tf.data.Dataset.from_tensor_slices(generator.match_train)
        train_match_dataset = train_match_dataset.shuffle(buffer_size=np.shape(generator.match_train)[0])
        train_match_dataset = train_match_dataset.repeat()
        train_match_dataset = train_match_dataset.batch(int(batch_size_train/2))
        
        train_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.no_match_train)
        train_non_match_dataset = train_non_match_dataset.shuffle(buffer_size=np.shape(generator.no_match_train)[0])
        train_non_match_dataset = train_non_match_dataset.repeat()
        train_non_match_dataset = train_non_match_dataset.batch(int((batch_size_train+1)/2))
        
        val_match_dataset_length = np.shape(generator.match_val)[0]
        val_match_dataset = tf.data.Dataset.from_tensor_slices(generator.match_val)
        val_match_dataset = val_match_dataset.shuffle(buffer_size = val_match_dataset_length)
        val_match_dataset = val_match_dataset.repeat()
        val_match_dataset = val_match_dataset.batch(batch_size_val)
        
        val_non_match_dataset_length = np.shape(generator.no_match_val)[0]
        val_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.no_match_val)
        val_non_match_dataset = val_non_match_dataset.shuffle(buffer_size = val_non_match_dataset_length)
        val_non_match_dataset = val_non_match_dataset.repeat()
        val_non_match_dataset = val_non_match_dataset.batch(batch_size_val)
        
        train_match_iterator = train_match_dataset.make_one_shot_iterator()
        val_match_iterator = val_match_dataset.make_one_shot_iterator()
        train_non_match_iterator = train_non_match_dataset.make_one_shot_iterator()
        val_non_match_iterator = val_non_match_dataset.make_one_shot_iterator()
        
        iterator = tf.data.Iterator.from_string_handle(handle, train_match_dataset.output_types)
        next_element = iterator.get_next()
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
   
    with tf.Session(config=config) as sess:
        if is_model_new:
            with tf.device(gpu_device_name):
                train_op = su.momentum_training(train_loss, learning_rate, momentum)
                sess.run(tf.global_variables_initializer()) # initialize all trainable parameters
                tf.add_to_collection("train_op",train_op)
        else:
            with tf.device(gpu_device_name):
                saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                train_op = tf.get_collection("train_op")[0]
                
        with tf.device(gpu_device_name): 
            graph = tf.get_default_graph()
            
            # Summary setup
            
            # get parameters of the first convolutional layer. Add filters and histograms
            # of filters and biases to summary
            conv0_filters = graph.get_tensor_by_name("conv1/kernel:0")
            nbr_of_filters_conv0 = sess.run(tf.shape(conv0_filters)[-1])
            hist_conv0 = tf.summary.histogram("hist_conv0", conv0_filters)
            conv0_filters = tf.transpose(conv0_filters, perm = [3,0,1,2])
            filter0 = tf.summary.image('Filter_0', conv0_filters, max_outputs=nbr_of_filters_conv0)
            conv0_bias = graph.get_tensor_by_name("conv1/bias:0")
            hist_bias0 = tf.summary.histogram("hist_bias0", conv0_bias)
            
            # get filters in first inception layer and their dimensions
            conv1_filters = graph.get_tensor_by_name("inception_1/conv1_1/kernel:0")
#            nbr_of_filters_conv1 = sess.run(tf.shape(conv1_filters)[-1])
            conv2_filters = graph.get_tensor_by_name("inception_1/conv2_1/kernel:0")
#            nbr_of_filters_conv2 = sess.run(tf.shape(conv2_filters)[-1])
            conv3_filters = graph.get_tensor_by_name("inception_1/conv3_1/kernel:0")
#            nbr_of_filters_conv3 = sess.run(tf.shape(conv3_filters)[-1])
            
            # histograms of filter weights
            hist_conv1 = tf.summary.histogram("hist_conv1", conv1_filters)
            hist_conv2 = tf.summary.histogram("hist_conv2", conv2_filters)
            hist_conv3 = tf.summary.histogram("hist_conv3", conv3_filters)
            
            # transpose filters to coincide with the dimensions requested by tensorflow's summary. 
            # Add filters to summary
#            conv1_filters = tf.transpose(conv1_filters, perm = [3,0,1,2])
#            filter1 = tf.summary.image('Filter_1', conv1_filters, max_outputs=nbr_of_filters_conv1)
#            conv2_filters = tf.transpose(conv2_filters, perm = [3,0,1,2])
#            filter2 = tf.summary.image('Filter_2', conv2_filters, max_outputs=nbr_of_filters_conv2)
#            conv3_filters = tf.transpose(conv3_filters, perm = [3,0,1,2])
#            filter3 = tf.summary.image('Filter_3', conv3_filters, max_outputs=nbr_of_filters_conv3)
            
            # get biases of filters in the first inception layer
            conv1_bias = graph.get_tensor_by_name("inception_1/conv1_1/bias:0")
            conv2_bias = graph.get_tensor_by_name("inception_1/conv2_1/bias:0")
            conv3_bias = graph.get_tensor_by_name("inception_1/conv3_1/bias:0")
            
            # histograms of filter biases
            hist_bias1 = tf.summary.histogram("hist_bias1", conv1_bias)
            hist_bias2 = tf.summary.histogram("hist_bias2", conv2_bias)
            hist_bias3 = tf.summary.histogram("hist_bias3", conv3_bias)
                
            summary_train_loss = tf.summary.scalar('training_loss', train_loss)
            
            summary_op = tf.summary.merge([summary_train_loss, filter0, hist_conv0, hist_bias0, hist_conv1, hist_bias1, hist_conv2, hist_bias2, hist_conv3, hist_bias3])
            train_writer = tf.summary.FileWriter(output_dir + "train_summary", graph=tf.get_default_graph())
             
            train_match_handle = sess.run(train_match_iterator.string_handle())
            val_match_handle = sess.run(val_match_iterator.string_handle())
            train_non_match_handle = sess.run(train_non_match_iterator.string_handle())
            val_non_match_handle = sess.run(val_non_match_iterator.string_handle())
            
        precision_over_time = []
        val_loss_over_time = []
        
        start_time_train = time.time()
        print("Starting training")
        # Training loop
        for i in range(1,train_itr + 1):
            if i % margin_itr == 0 and margin < max_margin:
                margin *= margin_factor
                
            train_batch_matching = sess.run(next_element,feed_dict={handle:train_match_handle})
            b_sim_train_matching = np.ones((np.shape(train_batch_matching)[0],1),dtype=np.int32)
            train_batch_non_matching = sess.run(next_element,feed_dict={handle:train_non_match_handle})
            b_sim_train_non_matching = np.zeros((np.shape(train_batch_non_matching)[0],1),dtype=np.int32)
            
            train_batch = np.append(train_batch_matching,train_batch_non_matching,axis=0)
            b_sim_train = np.append(b_sim_train_matching,b_sim_train_non_matching,axis=0)
            permutation = np.random.permutation(batch_size_train)
            train_batch = np.take(train_batch,permutation,axis=0)
            b_sim_train = np.take(b_sim_train,permutation,axis=0)
            
            # Randomize rotation of batch              
            rnd_rotation = np.random.randint(0,generator.rotation_res)
            b_l_train,b_r_train = generator.get_pairs(generator.train_data[rnd_rotation],train_batch)
            
            _,train_loss_value, summary = sess.run([train_op, train_loss, summary_op],feed_dict={left_train:b_l_train, right_train:b_r_train, label_train:b_sim_train, margin_holder:margin})

             # Use validation data set to tune hyperparameters (Classification threshold)
            if i % val_itr == 0:
                current_val_loss = 0
                b_sim_val_matching = np.repeat(np.ones((batch_size_val*int(val_match_dataset_length/batch_size_val),1)),generator.rotation_res,axis=0)
                b_sim_val_non_matching = np.repeat(np.zeros((batch_size_val*int(val_match_dataset_length/batch_size_val),1)),generator.rotation_res,axis=0)
                b_sim_full = np.append(b_sim_val_matching,b_sim_val_non_matching,axis=0)
                for j in range(int(val_match_dataset_length/batch_size_val)):
                    val_batch_matching = sess.run(next_element,feed_dict={handle:val_match_handle})
                    class_id_batch = generator.same_class(val_batch_matching)
                    for k in range(generator.rotation_res):
                        b_l_val,b_r_val = generator.get_pairs(generator.val_data[k],val_batch_matching) 
                        left_o,right_o,val_loss_value = sess.run([left_val_output,right_val_output, val_loss],feed_dict = {left_val:b_l_val, right_val:b_r_val, label_val:np.ones((batch_size_val,1)), margin_holder:margin})
                        current_val_loss += val_loss_value
                        if j == 0 and k == 0:
                            left_full = left_o
                            right_full = right_o
                            class_id = class_id_batch
                        else:
                            left_full = np.vstack((left_full,left_o))
                            right_full = np.vstack((right_full,right_o))
                            class_id = np.vstack((class_id, class_id_batch))
                    
                for j in range(int(val_match_dataset_length/batch_size_val)):
                    val_batch_non_matching = sess.run(next_element,feed_dict={handle:val_non_match_handle})
                    for k in range(generator.rotation_res):
                        b_l_val,b_r_val = generator.get_pairs(generator.val_data[0],val_batch_non_matching) 
                        left_o,right_o,val_loss_value  = sess.run([left_val_output,right_val_output,val_loss],feed_dict = {left_val:b_l_val, right_val:b_r_val, label_val:np.zeros((batch_size_val,1)), margin_holder:margin})
                        left_full = np.vstack((left_full,left_o))
                        right_full = np.vstack((right_full,right_o)) 
                        class_id_batch = generator.same_class(val_batch_non_matching)
                        class_id = np.vstack((class_id,class_id_batch))
                        current_val_loss += val_loss_value
                        
                val_loss_over_time.append(current_val_loss*batch_size_val/np.shape(b_sim_full)[0])
                precision, false_pos, false_neg, recall, fnr, fpr, inter_class_errors,_ = ie.get_test_diagnostics(left_full,right_full, b_sim_full,threshold, class_id)
            
                if false_pos > false_neg:   # Can use inter_class_errors to tune the threshold further
                    threshold -= thresh_step
                else:
                    threshold += thresh_step
                precision_over_time.append(precision)

            train_writer.add_summary(summary, i)
            
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
        plt.title("Validation loss over time")
        plt.xlabel("iteration")
        plt.ylabel("validation loss")
        plt.show()
        
if __name__ == "__main__":
     main(sys.argv[1:])