# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:34:36 2018

@author: Tuong Lam
"""

import numpy as np
import scipy.linalg as sl
import tensorflow as tf
import os 
import utilities as util
import pickle

def get_test_diagnostics(left_pairs_o,right_pairs_o,sim_labels,threshold):
    """ Computes and returns evaluation metrics from testing.
    
    Input:
    left_pairs_o - numpy array with rows corresponding to arrays obtained from inference step in the siamese network
    right_pairs_o - numpy array with rows corresponding to arrays obtained from inference step in the siamese network
    sim_labels - ground truth for pairs of arrays (1 if the arrays correspond to matching images, 0 otherwise)
    threshold - distance threshold, if the 2-norm distanc between two arrays are less than or equal to this value 
    they are considered to correspond to a matching pair of images.
    Returns:
    precision - precision
    false_pos - number of false positives
    false_neg - number of false negatives
    recall - recall (nbr of true positives/total number of positive examples)
    fnr - false negative rate (false negative/total number of positive examples)
    fpr - false positive rate (false positive/total number of negative examples)
    """
    matching = np.zeros(len(sim_labels),dtype=np.int32)
    l2_normalized_diff = util.l2_normalize(left_pairs_o-right_pairs_o)
    false_pos = 0
    false_neg = 0
    p = np.sum(sim_labels)
    n = len(sim_labels) - p
    for i in range(len(sim_labels)):
#        print(sl.norm(l2_normalized_diff[i,:]))
        if sl.norm(l2_normalized_diff[i,:]) < threshold:
            matching[i] = 1
            if sim_labels[i] == 0:
                false_pos = false_pos + 1
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
    
    return precision, false_pos, false_neg, recall, fnr, fpr
 
def evaluate_siamese_network(generator,batch_size_test,threshold,output_dir):
    """ This method is used to evaluate a siamese network for mnist dataset.
    
    The model is defined in the file siamese_nn_model_mnist.py and trained in 
    the file siamese_nn_train_mnist.py. Evaluation will only be performed if
    a model exists. The method will print evaluation metrics.
    
    Input:
    generator - an instance of a data_generator object used in training
    batch_size_test - batch size for the test placeholder
    threshold - distance threshold (2-norm) for the decision stage
    output_dir - the directory of the siamese model
    """
    
    tf.reset_default_graph()
    
    if not os.path.exists(output_dir + ".meta"):
        print("No siamese model exists in " + output_dir)
        return
        
    else:
        print("Using existing model in the directory " + output_dir + " for evaluation")        
        saver = tf.train.import_meta_graph(output_dir + ".meta")        
        g = tf.get_default_graph()
        left_test = g.get_tensor_by_name("left_test:0")
        right_test = g.get_tensor_by_name("right_test:0")
        
        left_test_inference = tf.get_collection("left_test_output")[0]
        right_test_inference = tf.get_collection("right_test_output")[0]
        
        handle= g.get_tensor_by_name("handle:0")
        
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(output_dir))
            
            test_match_dataset = tf.data.Dataset.from_tensor_slices(generator.all_match_test)
            test_match_dataset = test_match_dataset.batch(batch_size_test)
            test_match_dataset_length = np.shape(generator.all_match_test)[0]
        
#            test_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.all_non_match_test)
            test_non_match_dataset_length = np.shape(generator.all_non_match_test)[0]
            test_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.all_non_match_val[0:int(test_non_match_dataset_length/10)])
            test_non_match_dataset = test_non_match_dataset.batch(batch_size_test)
            test_non_match_dataset_length = np.shape(generator.all_non_match_test)[0]
            
            test_match_iterator = test_match_dataset.make_one_shot_iterator()
            test_match_handle = sess.run(test_match_iterator.string_handle())
        
            test_non_match_iterator = test_non_match_dataset.make_one_shot_iterator()
            test_non_match_handle = sess.run(test_non_match_iterator.string_handle())
            
            iterator = tf.data.Iterator.from_string_handle(handle, test_match_dataset.output_types)
            next_element = iterator.get_next()
            
            sim_full = np.vstack((np.ones((batch_size_test*int(test_match_dataset_length/batch_size_test),1)),np.zeros((batch_size_test*int((int(test_non_match_dataset_length/10))/batch_size_test),1))))
#            sim_full = np.vstack((np.ones((batch_size_test*int(test_match_dataset_length/batch_size_test),1)),np.zeros((batch_size_test*int(int(test_non_match_dataset_length/10)/batch_size_test),1))))

            for i in range(int(test_match_dataset_length/batch_size_test)):
                test_batch = sess.run(next_element,feed_dict={handle:test_match_handle})
                b_l_test,b_r_test = generator.get_pairs(generator.test_data,test_batch) 
                left_o,right_o = sess.run([left_test_inference,right_test_inference],feed_dict = {left_test:b_l_test, right_test:b_r_test})
                if i == 0:
                    left_full = left_o
                    right_full = right_o
                else:
                    left_full = np.vstack((left_full,left_o))
                    right_full = np.vstack((right_full,right_o))
                    
            for i in range(int(int(test_non_match_dataset_length/10)/batch_size_test)):
                test_batch = sess.run(next_element,feed_dict={handle:test_non_match_handle})
                b_l_test,b_r_test = generator.get_pairs(generator.test_data,test_batch) 
                left_o,right_o = sess.run([left_test_inference,right_test_inference],feed_dict = {left_test:b_l_test, right_test:b_r_test})
                left_full = np.vstack((left_full,left_o))
                right_full = np.vstack((right_full,right_o))     
            
            precision, false_pos, false_neg, recall, fnr, fpr = get_test_diagnostics(left_full,right_full,sim_full,threshold)

            print("Precision: %f " % precision)
            print("# False positive: %d " % false_pos)
            print("# False negative: %d " % false_neg)
            print("# Recall: %f " % recall)
            print("# Miss rate/false negative rate: %f " % fnr)
            print("# fall-out/false positive rate: %f " % fpr)
    
    
def main(unused_argv):
   """ Runs evaluation on mnist siamese network
   
   """
    
   # Set parameters for evaluation
   threshold = 0.475
   batch_size = 1000
    
   output_dir = "/tmp/siamese_mnist_model/"
    
    # Load generator
   with open('generator_data.pk1', 'rb') as input:
       generator = pickle.load(input)
    
   evaluate_siamese_network(generator, batch_size, threshold, output_dir)
    
if __name__ == "__main__":
    tf.app.run()