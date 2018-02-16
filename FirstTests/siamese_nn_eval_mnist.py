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
from data_generator import data_generator
import pickle


def get_eval_diagnostics(left_pairs_o,right_pairs_o,sim_labels,threshold):
    """ Computes and returns evaluation metrics.
    
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
    matching = np.zeros(len(sim_labels))
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
 
def evaluate_siamese_network(generator, nbr_of_eval_pairs, eval_itr, threshold,output_dir):
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
    
    if not os.path.exists(output_dir + ".meta"):
        print("No siamese model exists in " + output_dir)
        return
        
    else:
        print("Using existing model in the directory " + output_dir + " for evaluation")        
        saver = tf.train.import_meta_graph(output_dir + ".meta")        
        g = tf.get_default_graph()
        left_eval = g.get_tensor_by_name("left_eval:0")
        right_eval = g.get_tensor_by_name("right_eval:0")
        
        left_eval_inference = tf.get_collection("left_eval_output")[0]
        right_eval_inference = tf.get_collection("right_eval_output")[0]
        
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(output_dir))
            for i in range(eval_itr):
                left,right,sim = generator.gen_eval_batch(nbr_of_eval_pairs)
                left_o,right_o= sess.run([left_eval_inference,right_eval_inference],feed_dict = {left_eval:left, right_eval:right})
                if i == 0:
                    left_full = left_o
                    right_full = right_o
                    sim_full = sim
                else:
                    left_full = np.vstack((left_full,left_o))
                    right_full = np.vstack((right_full,right_o))
                    sim_full = np.vstack((sim_full, sim))
                
            precision, false_pos, false_neg, recall, fnr, fpr = get_eval_diagnostics(left_full,right_full,sim_full,threshold)

            print("Precision: %f " % precision)
            print("# False positive: %d " % false_pos)
            print("# False negative: %d " % false_neg)
            print("# Recall: %f " % recall)
            print("# Miss rate/false negative rate: %f " % fnr)
            print("# fall-out/false positive rate: %f " % fpr)
    


def main(unused_argv):
    ''' Runs evaluation on mnist's evaluation data set '''
    
    # Set parameters for evaluation
    eval_itr = 1
    threshold = 0.45
    nbr_of_images = 10000
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir = "/tmp/siamese_mnist_model/"
    #Load Evaluation data and set up generator
    if not os.path.exists(dir_path + "/generator_data_eval.pk1"):
        with open('generator_data_eval.pk1', 'wb') as output:
            # Load mnist training and eval data and perform necessary data reshape
            mnist = tf.contrib.learn.datasets.load_dataset("mnist")
            eval_data = util.reshape_grayscale_data(mnist.test.images) # Returns np.array
            eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
            generator = data_generator(eval_data,eval_labels,nbr_of_images) # initialize data generator
            pickle.dump(generator, output, pickle.HIGHEST_PROTOCOL)
    else:
        # Load generator
        with open('generator_data_eval.pk1', 'rb') as input:
            generator = pickle.load(input)
    
    evaluate_siamese_network(generator, nbr_of_images, eval_itr, threshold, output_dir)
    
    
    
if __name__ == "__main__":
    tf.app.run()
    
    
    
    
    
    
    
    
    
    

        