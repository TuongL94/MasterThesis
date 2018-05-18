# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:34:36 2018

@author: Tuong Lam
"""

import numpy as np
import tensorflow as tf
import os 
import sys
import pickle
import scipy.linalg as sl

# imports from self-implemented modules
import capsule_utility as cu
import utilities as util

def get_test_diagnostics(left_pairs_o,right_pairs_o,sim_labels,threshold,class_id=None,plot_hist=False, breakpoint=None):
    """ Computes and returns evaluation metrics.
    
    Input:
    left_pairs_o - numpy array with first dimension corresponding to an output from the capsule network
    right_pairs_o - numpy array with first dimension corresponding to an output from the capsule network
    sim_labels - ground truth for pairs of images (1 if the images correspond to matching images, 0 otherwise)
    threshold - distance threshold, if the 2-norm distance between two grid features are less than or equal to this value 
    they are considered to correspond to a matching pair of images.
    class_id - Is optional. Contains information about which finger and person each fingerprint comes from.
    Returns:
    precision - precision
    false_pos - number of false positives
    false_neg - number of false negatives
    recall - recall (nbr of true positives/total number of positive examples)
    fnr - false negative rate (false negative/total number of positive examples)
    fpr - false positive rate (false positive/total number of negative examples)
    inter_class_errors - number of false positive from the same finger+person (class)
    """
    matching = np.zeros(len(sim_labels))
#    l2_distances = cu.grid_wise_norm(left_pairs_o, right_pairs_o)
    l2_distances = np.squeeze(sl.norm(left_pairs_o - right_pairs_o, ord = 2, axis=-2))
    mean_l2_dist = np.mean(l2_distances, axis=1)
    
    if plot_hist:
        util.get_separation_distance_hist(mean_l2_dist[0:breakpoint],mean_l2_dist[breakpoint:])
    
    false_pos = 0
    false_neg = 0
    inter_class_errors = 0
    p = np.sum(sim_labels)
    n = len(sim_labels) - p
    for i in range(len(sim_labels)):
        if mean_l2_dist[i] < threshold:
            matching[i] = 1
            if sim_labels[i] == 0:
                false_pos = false_pos + 1
                if not class_id is None:
                    if class_id[i] == 1:
                        inter_class_errors += 1
        else:
            if sim_labels[i] == 1:
                false_neg = false_neg + 1
    
    accuracy = np.sum((matching == sim_labels.T))/len(sim_labels)
    tp = 0
    tn = 0
    for i in range(len(sim_labels)):
        if matching[i] == 1 and sim_labels[i] == 1:
            tp += 1
        elif matching[i] == 0 and sim_labels[i] == 0:
            tn += 1
    recall = tp/p
    tnr = tn/n
    fnr = 1 - recall
    fpr = false_pos/n
    
    return accuracy, false_pos, false_neg, recall, fnr, fpr, tnr, inter_class_errors

 
def evaluate_capsule_network(generator, batch_size, thresholds, eval_itr, output_dir, gpu_device_name, negative_multiplier, matrics_name, metrics_path):
    """ This method is used to evaluate a capsule network for fingerprint datasets.
    
    The model is defined in the file capsule_nn_model.py and trained in 
    the file capsule_nn_train.py. Evaluation will only be performed if
    a model exists. The method will print evaluation metrics.
    
    Input:
    generator - an instance of a data_generator object used in training
    batch_size - batch size for the evaluation placeholder
    threshold - distance threshold (2-norm) for the decision stage
    eval_itr - number of evaluation iterations
    output_dir - the directory of the trained model
    gpu_device_name - name of the GPU device to run the evaluation with
    """
    
    tf.reset_default_graph()
    
    if not os.path.exists(output_dir + "checkpoint"):
        print("No model exists in " + output_dir)
        return
        
    else:
        print("Using existing model in the directory " + output_dir + " for evaluation")  
        for file in os.listdir(output_dir):
            if file.endswith(".meta"):
                meta_file_name = os.path.join(output_dir,file)
        saver = tf.train.import_meta_graph(meta_file_name)
        
        with tf.device(gpu_device_name):
            g = tf.get_default_graph()
#            left_test = g.get_tensor_by_name("left_test:0")
#            right_test = g.get_tensor_by_name("right_test:0")
            
#            left_test_inference = tf.get_collection("left_test_output")[0]
#            right_test_inference = tf.get_collection("right_test_output")[0]
            
            left_test = g.get_tensor_by_name("left_image_holder_test:0")
            right_test = g.get_tensor_by_name("right_image_holder_test:0")
            
            left_test_inference = tf.get_collection("left_test_output")[0]
            right_test_inference = tf.get_collection("right_test_output")[0]
            
            handle = g.get_tensor_by_name("handle:0")
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            with tf.device(gpu_device_name):
                saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                
                test_match_dataset = tf.data.Dataset.from_tensor_slices(generator.match_test)
                test_match_dataset = test_match_dataset.batch(batch_size)
#                test_match_dataset_length = np.shape(generator.match_test)[0]
            
#                test_non_match_dataset_length = np.shape(generator.no_match_test)[0]
                test_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.no_match_test)
#                test_non_match_dataset = test_non_match_dataset.shuffle(buffer_size = test_non_match_dataset_length)
                test_non_match_dataset = test_non_match_dataset.batch(batch_size)
                
                test_match_iterator = test_match_dataset.make_one_shot_iterator()
                test_match_handle = sess.run(test_match_iterator.string_handle())
            
                test_non_match_iterator = test_non_match_dataset.make_one_shot_iterator()
                test_non_match_handle = sess.run(test_non_match_iterator.string_handle())
                
                iterator = tf.data.Iterator.from_string_handle(handle, test_match_dataset.output_types)
                next_element = iterator.get_next()
                
#                sim_full = np.vstack((np.ones((batch_size*eval_itr,1)), np.zeros((batch_size*eval_itr,1))))
                breakpoint = batch_size*eval_itr
                sim_full = np.vstack((np.ones((breakpoint,1)), np.zeros((negative_multiplier*breakpoint,1))))
                
                for i in range(eval_itr):
                    test_batch = sess.run(next_element,feed_dict={handle:test_match_handle})
                    for j in range(generator.rotation_res):
                        b_l_test,b_r_test = generator.get_pairs(generator.test_data[j],test_batch)
#                        class_id_batch = generator.same_class(test_batch,test=True)
                        left_o,right_o = sess.run([left_test_inference,right_test_inference],feed_dict = {left_test:b_l_test, right_test:b_r_test})
                        if i == 0 and j == 0:
                            left_full = left_o
                            right_full = right_o
#                            class_id = class_id_batch
                        else:
                            left_full = np.vstack((left_full,left_o))
                            right_full = np.vstack((right_full,right_o))
#                            class_id = np.vstack((class_id, class_id_batch))
    
                for i in range(eval_itr * negative_multiplier):
                    test_batch = sess.run(next_element,feed_dict={handle:test_non_match_handle})
                    for j in range(generator.rotation_res):
                        b_l_test,b_r_test = generator.get_pairs(generator.test_data[j],test_batch) 
                        left_o,right_o = sess.run([left_test_inference,right_test_inference],feed_dict = {left_test:b_l_test, right_test:b_r_test})
                        left_full = np.vstack((left_full,left_o))
                        right_full = np.vstack((right_full,right_o))   
                        
#                        class_id_batch = generator.same_class(test_batch,test=True)
#                        class_id = np.vstack((class_id, class_id_batch))

                for i in range(len(thresholds)):
                    
                    accuracy, false_pos, false_neg, recall, fnr, fpr, tnr, _ = get_test_diagnostics(left_full,right_full,sim_full,thresholds[i])
                    metrics = (fpr, fnr, recall, accuracy)
                    # save evaluation metrics to a file 
                    util.save_evaluation_metrics(metrics, metrics_path + matrics_name)
                
                
                accuracy, false_pos, false_neg, recall, fnr, fpr,tnr, _ = get_test_diagnostics(left_full,right_full,sim_full,0.05,plot_hist=True, breakpoint = breakpoint)
    
                print("Accuracy: %f " % accuracy)
                print("# False positive: %d " % false_pos)
                print("# False negative: %d " % false_neg)
#                print("# Number of false positive from the same class: %d " % inter_class_errors)
                print("# Recall: %f " % recall)
                print("# Miss rate/false negative rate: %f " % fnr)
                print("# fall-out/false positive rate: %f " % fpr)
                      
#                nbr_same_class = np.sum(class_id[eval_itr*batch_size:])
#                print("Number of fingerprints in the same class in the non matching set: %d " % nbr_same_class)
                
                  # get evaluation metrics for varying thresholds
                fpr_vals, fnr_vals, recall_vals, acc_vals = util.get_evaluation_metrics_vals(metrics_path + matrics_name)
    
                # plots of evaluation metrics
                util.plot_evaluation_metrics(thresholds, fpr_vals, fnr_vals, recall_vals, acc_vals)
         
def main(argv):
    """ Runs evaluation on trained network 
    """
    # Set parameters for evaluation
    thresholds = np.linspace(0, 0.05, num=100)
    batch_size = 10
    eval_itr = 10
    negative_multiplier = 1
    metrics_name = "10.txt"
    
    output_dir = argv[0] # directory where the model is saved
    data_path =  argv[1]
    metrics_path = argv[0]
    gpu_device_name = argv[-1] 
   
    # if file containing evaluation metrics already exists use this data directly
#    if os.path.exists(metrics_path + metrics_name):
#        # get evaluation metrics for varying thresholds
#        fpr_vals, fnr_vals, recall_vals, acc_vals = util.get_evaluation_metrics_vals(metrics_path + metrics_name)
#
#        # plots of evaluation metrics
#        util.plot_evaluation_metrics(thresholds, fpr_vals, fnr_vals, recall_vals, acc_vals)
#        return
    
    # Load generator
    with open(data_path + "generator_data_small_rotdiff5_transdiff10_new.pk1", "rb") as input:
        generator = pickle.load(input)
    
    evaluate_capsule_network(generator, batch_size, thresholds, eval_itr, output_dir, gpu_device_name, negative_multiplier, metrics_name, metrics_path)
    
if __name__ == "__main__":
    main(sys.argv[1:])