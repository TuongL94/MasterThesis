# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:34:36 2018

@author: Tuong Lam
"""

import numpy as np
import scipy.linalg as sl
import tensorflow as tf
import os 
import sys
import utilities as util
import pickle
import re

def get_test_diagnostics(left_pairs_o,right_pairs_o,sim_labels,threshold,class_id=None):
    """ Computes and returns evaluation metrics.
    
    Input:
    left_pairs_o - numpy array with rows corresponding to arrays obtained from inference step in the siamese network
    right_pairs_o - numpy array with rows corresponding to arrays obtained from inference step in the siamese network
    sim_labels - ground truth for pairs of arrays (1 if the arrays correspond to matching images, 0 otherwise)
    threshold - distance threshold, if the 2-norm distanc between two arrays are less than or equal to this value 
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
    l2_normalized_diff = util.l2_normalize(left_pairs_o-right_pairs_o)
    false_pos = 0
    false_neg = 0
    inter_class_errors = 0
    p = np.sum(sim_labels)
    n = len(sim_labels) - p
    for i in range(len(sim_labels)):
#        print(sl.norm(l2_normalized_diff[i,:]))
        if np.isinf(l2_normalized_diff[i,:]).any() or np.isnan(l2_normalized_diff[i,:]).any():
            print('Got inf or Nan in L2 norm; Change hyperparameters to avoid')
            if sim_labels[i] == 1:
                false_neg = false_neg + 1
        elif sl.norm(l2_normalized_diff[i,:]) < threshold:
            matching[i] = 1
            if sim_labels[i] == 0:
                false_pos = false_pos + 1
                if not class_id is None:
                    if class_id[i] == 1:
                        inter_class_errors += 1
        else:
            if sim_labels[i] == 1:
                false_neg = false_neg + 1
    
    precision = np.sum((matching == sim_labels.T))/len(sim_labels)
    tp = 0
    for i in range(len(sim_labels)):
        if matching[i] == 1 and sim_labels[i] == 1:
            tp += 1
    recall = tp/p
    fnr = 1 - recall
    fpr = false_pos/n
    
    return precision, false_pos, false_neg, recall, fnr, fpr#, inter_class_errors
 
def evaluate_siamese_network(generator, batch_size, threshold, output_dir, eval_itr,gpu_device_name):
    """ This method is used to evaluate a siamese network for fingerprint datasets.
    
    The model is defined in the file siamese_nn_model.py and trained in 
    the file siamese_nn_train.py. Evaluation will only be performed if
    a model exists. The method will print evaluation metrics.
    
    Input:
    generator - an instance of a data_generator object used in training
    nbr_of_eval_pairs - batch size for the evaluation placeholder
    eval_itr - number of evaluation iterations
    threshold - distance threshold (2-norm) for the decision stage
    output_dir - the directory of the siamese model
    """
    
    tf.reset_default_graph()
    
    if not os.path.exists(output_dir + "checkpoint"):
        print("No siamese model exists in " + output_dir)
        return
        
    else:
        print("Using existing model in the directory " + output_dir + " for evaluation")  
        with open(output_dir + "checkpoint","r") as file:
            line  = file.readline()
            words = re.split("/",line)
            model_file_name = words[-1][:-2]
            saver = tf.train.import_meta_graph(output_dir + model_file_name + ".meta")
        
        with tf.device(gpu_device_name):
            g = tf.get_default_graph()
            left_test = g.get_tensor_by_name("left_test:0")
            right_test = g.get_tensor_by_name("right_test:0")
            
            left_test_inference = tf.get_collection("left_test_output")[0]
            right_test_inference = tf.get_collection("right_test_output")[0]
            
            handle= g.get_tensor_by_name("handle:0")
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device(gpu_device_name):
                saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                
                test_anchors_dataset = tf.data.Dataset.from_tensor_slices(generator.anchors_test)
#                test_anchors_dataset = train_anchors_dataset.shuffle(buffer_size=np.shape(generator.anchors_test)[0])
#                test_anchors_dataset = train_anchors_dataset.repeat()
                test_anchors_dataset = test_anchors_dataset.batch(batch_size)
#                test_anchors_dataset_length = np.shape(generator.anchors_test)[0]
            
                test_anchors_iterator = test_anchors_dataset.make_one_shot_iterator()
                test_anchors_handle = sess.run(test_anchors_iterator.string_handle())
            
                iterator = tf.data.Iterator.from_string_handle(handle, test_anchors_dataset.output_types)
                next_element = iterator.get_next()
                
    #            sim_full = np.vstack((np.ones((batch_size_test*int(test_match_dataset_length/batch_size_test),1)),np.zeros((batch_size_test*int(test_non_match_dataset_length/batch_size_test),1))))
                labels = np.vstack((np.ones((batch_size,1)), np.zeros((batch_size,1))))
    #            sim_full = np.vstack((np.ones((batch_size_test*int(test_match_dataset_length/batch_size_test),1)),np.zeros((batch_size_test*int((int(test_non_match_dataset_length/10))/batch_size_test),1))))
    #            sim_full = np.vstack((np.ones((batch_size_test*int(test_match_dataset_length/batch_size_test),1)),np.zeros((batch_size_test*int(int(test_non_match_dataset_length/10)/batch_size_test),1))))
                
                for i in range(eval_itr):
                    #            for i in range(int(test_match_dataset_length/batch_size_test)):
                    test_batch_anchors = sess.run(next_element,feed_dict={handle:test_anchors_handle})
    
                    for j in range(generator.rotation_res):
                        b_anch_test,b_pos_test,b_neg_test= generator.get_triplet(generator.test_data[j], generator.triplets_test, test_batch_anchors)
#                        class_id_batch = generator.same_class(test_batch,test=True)
                        left_o_match,right_o_match = sess.run([left_test_inference,right_test_inference],feed_dict = {left_test:b_anch_test, right_test:b_pos_test})
                        left_o_no_match,right_o_no_match = sess.run([left_test_inference,right_test_inference],feed_dict = {left_test:b_anch_test, right_test:b_neg_test})
                        if i == 0 and j == 0:
                            left_full = np.vstack((left_o_match,left_o_no_match))
                            right_full = np.vstack((right_o_match,right_o_no_match))
                            labels_full = labels
#                            class_id = class_id_batch
                        else:
                            left_full = np.vstack((left_full,left_o_match,left_o_no_match))
                            right_full = np.vstack((right_full,right_o_match,right_o_no_match))
                            labels_full = np.vstack((labels_full,labels))
#                            class_id = np.vstack((class_id, class_id_batch))
    
#                for i in range(eval_itr):
#    #            for i in range(int(int(test_non_match_dataset_length/10)/batch_size_test)):
#                    test_batch = sess.run(next_element,feed_dict={handle:test_non_match_handle})
#                    for j in range(generator.rotation_res):
#                        b_l_test,b_r_test = generator.get_pairs(generator.test_data[j],test_batch) 
#                        left_o,right_o = sess.run([left_test_inference,right_test_inference],feed_dict = {left_test:b_l_test, right_test:b_r_test})
#                        left_full = np.vstack((left_full,left_o))
#                        right_full = np.vstack((right_full,right_o))   
#                        
#                        class_id_batch = generator.same_class(test_batch,test=True)
#                        class_id = np.vstack((class_id, class_id_batch))

                
                precision, false_pos, false_neg, recall, fnr, fpr = get_test_diagnostics(left_full,right_full,labels_full,threshold)
    
                print("Precision: %f " % precision)
                print("# False positive: %d " % false_pos)
                print("# False negative: %d " % false_neg)
#                print("# Number of false positive from the same class: %d " % inter_class_errors)
                print("# Recall: %f " % recall)
                print("# Miss rate/false negative rate: %f " % fnr)
                print("# fall-out/false positive rate: %f " % fpr)
                      
#                nbr_same_class = np.sum(class_id[eval_itr*batch_size:])
#                print("Number of fingerprints in the same class in the non matching set: %d " % nbr_same_class)
         
def main(argv):
   """ Runs evaluation on mnist siamese network"""
    
    # Set parameters for evaluation
   threshold = 0.5
   batch_size = 100
   eval_itr = 1
    
   dir_path = os.path.dirname(os.path.realpath(__file__))
   output_dir = dir_path + "/train_models" + argv[0] + "/" # directory where the model is saved
   gpu_device_name = argv[-1] 
   
    # Load generator
   with open('generator_data.pk1', 'rb') as input:
       generator = pickle.load(input)
    
   evaluate_siamese_network(generator, batch_size, threshold, output_dir, eval_itr, gpu_device_name)
    
if __name__ == "__main__":
    main(sys.argv[1:])