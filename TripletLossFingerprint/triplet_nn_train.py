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

# imports from self-implemented modules
import utilities as util
import triplet_nn_eval as tre
import triplet_nn_model as tr

def main(argv):
    """ This method is used to train a triplet network for fingerprint datasets.
    
    The model is defined in the file triplet_nn_model.py.
    If a model exists it will be used for further training, otherwise a new
    one is created. It is also possible to evaluate the model directly after training.
    
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
    if not os.path.exists(data_path + "generator_triplet_data.pk1"):
        with open(data_path + "generator_triplet_data.pk1", "wb") as output:
            # Load fingerprint labels and data from file with names
            finger_id = np.load(data_path + "/finger_id.npy")
            person_id = np.load(data_path + "/person_id.npy")
            finger_data = np.load(data_path + "/fingerprints.npy")
            translation = np.load(data_path + "/translation.npy")
            rotation = np.load(data_path + "/rotation.npy")
            nbr_of_images = np.shape(finger_data)[0] # number of images to use from the original data set
            finger_data = util.reshape_grayscale_data(finger_data)
            rotation_res = 1
            
            generator = data_generator(finger_data, finger_id, person_id, translation, rotation, nbr_of_images, rotation_res) # initialize data generator
            
            finger_id_gt_vt = np.load(data_path+ "/finger_id_gt_vt.npy")
            person_id_gt_vt = np.load(data_path + "/person_id_gt_vt.npy")
            finger_data_gt_vt = np.load(data_path + "/fingerprints_gt_vt.npy")
            translation_gt_vt = np.load(data_path + "/translation_gt_vt.npy")
            rotation_gt_vt = np.load(data_path + "/rotation_gt_vt.npy")
            
            finger_data_gt_vt = util.reshape_grayscale_data(finger_data_gt_vt)
            nbr_of_images_gt_vt = np.shape(finger_data_gt_vt)[0]          
            
            generator.add_new_data(finger_data_gt_vt, finger_id_gt_vt, person_id_gt_vt, translation_gt_vt, rotation_gt_vt, nbr_of_images_gt_vt)
            pickle.dump(generator, output, pickle.HIGHEST_PROTOCOL)
    else:
        # Load generator
        with open(data_path + "generator_triplet_data.pk1", "rb") as input:
            generator = pickle.load(input)
             
    # parameters for training
    batch_size_train = 300
    train_itr = 1000000000000
    learning_rate = 0.00001
    momentum = 0.99
    
    # margin setup for triplet loss
    margin =  4.0 
    margin_factor = 1.1
    max_margin = 1.25 # maximum allowed margin value
    margin_itr = 5000 # frequency in which to increase the margin in loss function
    
    # Paramerters for increasing difficulty
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
            anchor_train,pos_train,neg_train,anchor_val,pos_val,neg_val,left_test,right_test = tr.placeholder_inputs(image_dims)
            handle = tf.placeholder(tf.string, shape=[],name="handle")
            margin_holder = tf.placeholder(tf.float32, shape=[], name="margin_holder")

            anchor_train_output = tr.inference(anchor_train)            
            pos_train_output = tr.inference(pos_train)
            neg_train_output = tr.inference(neg_train)
            anchor_val_output = tr.inference(anchor_val)
            pos_val_output = tr.inference(pos_val)
            neg_val_output = tr.inference(neg_val)
            
            left_test_output = tr.inference(left_test)
            right_test_output = tr.inference(right_test)
            
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            train_loss = tr.triplet_loss(anchor_train_output,pos_train_output,neg_train_output,margin_holder)
            
            # add regularization terms to loss function
            for i in range(len(reg_losses)):
                train_loss += reg_losses[i]
            
            val_loss = tr.triplet_loss(anchor_val_output,pos_val_output,neg_val_output,margin_holder)
            
            tf.add_to_collection("train_loss",train_loss)
            tf.add_to_collection("val_loss",val_loss)
            tf.add_to_collection("anchor_val_output",anchor_val_output)
            tf.add_to_collection("pos_val_output",pos_val_output)
            tf.add_to_collection("neg_val_output",neg_val_output)
            tf.add_to_collection("left_test_output",left_test_output)
            tf.add_to_collection("right_test_output",right_test_output)
                            
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
            margin_holder = g.get_tensor_by_name("margin_holder:0")
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
#    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    with tf.Session(config=config) as sess:
        if is_model_new:
            with tf.device(gpu_device_name):
                train_op = tr.training(train_loss, learning_rate, momentum)
                sess.run(tf.global_variables_initializer()) # initialize all trainable parameters
                tf.add_to_collection("train_op",train_op)
        else:
            with tf.device(gpu_device_name):
                saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                train_op = tf.get_collection("train_op")[0]
            
#            for i in sess.graph.get_operations():
#                print(i.values())
                
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
            train_anchors_dataset = train_anchors_dataset.batch(batch_size_train)
            
            val_anchors_dataset_length = np.shape(generator.anchors_val)[0]
            val_anchors_dataset = tf.data.Dataset.from_tensor_slices(generator.anchors_val)
            val_anchors_dataset = val_anchors_dataset.shuffle(buffer_size = val_anchors_dataset_length)
            val_anchors_dataset = val_anchors_dataset.repeat()
            val_anchors_dataset = val_anchors_dataset.batch(batch_size_val)
                   
            train_anchors_iterator = train_anchors_dataset.make_one_shot_iterator()
            train_anchors_handle = sess.run(train_anchors_iterator.string_handle())
            
            val_anchors_iterator = val_anchors_dataset.make_one_shot_iterator()
            val_anchors_handle = sess.run(val_anchors_iterator.string_handle())
            
            iterator = tf.data.Iterator.from_string_handle(handle, train_anchors_dataset.output_types)
            next_element = iterator.get_next()
        
        print("Starting training")
        start_time_train = time.time()
        # Training loop
        for i in range(1,train_itr + 1):
            
            if i % margin_itr == 0 and margin < max_margin:
                margin *= margin_factor
                
            train_batch_anchors = sess.run(next_element,feed_dict={handle:train_anchors_handle})
            
            ####### Increase difficulty every harder_itr iteration by offline evaluation on a subset of all triplets #######
            if i % harder_itr == 0:                   
                # Take a random subset of each anchors non matching set
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
                
            # Randomize rotation of batch              
            rnd_rotation = np.random.randint(0,generator.rotation_res)
            b_anch_train,b_pos_train,b_neg_train = generator.get_triplet(generator.train_data[rnd_rotation], generator.triplets_train, train_batch_anchors)
            
            _,train_loss_value, summary = sess.run([train_op, train_loss, summary_op],feed_dict={anchor_train:b_anch_train, pos_train:b_pos_train, neg_train:b_neg_train, margin_holder:margin})

             # Use validation data set to tune hyperparameters (threshold)
            if i % val_itr == 0:
                current_val_loss = 0
                labels = np.vstack((np.ones((batch_size_val,1)), np.zeros((batch_size_val,1))))
                for j in range(int(val_anchors_dataset_length/batch_size_val)):
                    val_batch_anchors = sess.run(next_element,feed_dict={handle:val_anchors_handle})
                    for k in range(generator.rotation_res):
                        b_anch_val,b_pos_val,b_neg_val= generator.get_triplet(generator.val_data[k], generator.triplets_val, val_batch_anchors)
                        anchor_o,pos_o,neg_o,val_loss_value = sess.run([anchor_val_output,pos_val_output, neg_val_output,val_loss],feed_dict = {anchor_val:b_anch_val, pos_val:b_pos_val, neg_val:b_neg_val, margin_holder:margin})
                        current_val_loss += val_loss_value
                        if j == 0 and k == 0:
                            anchor_full = np.vstack((anchor_o,anchor_o))
                            candidates_full = np.vstack((pos_o,neg_o))
                            labels_full = labels
                        else:
                            anchor_full = np.vstack((anchor_full,anchor_o,anchor_o))
                            candidates_full = np.vstack((candidates_full,pos_o,neg_o))
                            labels_full = np.vstack((labels_full,labels))
                                            
                val_loss_over_time.append(current_val_loss*batch_size_val/np.shape(labels_full)[0])
                precision, false_pos, false_neg, recall, fnr, fpr, _ = tre.get_test_diagnostics(anchor_full,candidates_full, labels_full,threshold)
            
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