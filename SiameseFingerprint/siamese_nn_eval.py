# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:34:36 2018

@author: Tuong Lam
"""
from data_generator import data_generator
import numpy as np
import scipy.linalg as sl
import tensorflow as tf
import os 
import utilities as util

def evaluate_siamese_network(left_pairs_o,right_pairs_o,sim_labels,threshold):
    """ Computes evaluation metrics.
    
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

def main(unused_argv):
    """ This method is used to evaluate a siamese network for fingerprint datasets.
    
    The model is defined in the file siamese_nn_model.py and trained in 
    the file siamese_nn_train.py. Evaluation will only be performed if
    a model exists.
    
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    eval_finger = np.load(dir_path + "/finger_id.npy")
    eval_person = np.load(dir_path + "/person_id.npy")
    eval_data = np.load(dir_path + "/fingerprints.npy")    
    translation = np.load(dir_path + "/translation.npy")
    rotation = np.load(dir_path + "/rotation.npy")
    
    nbr_of_training_images = np.shape(eval_data)[0] # number of images to use from the training data set
    
    eval_data = util.reshape_grayscale_data(eval_data)
    generator = data_generator(eval_data, eval_finger, eval_person, translation, rotation, nbr_of_training_images) # initialize data generator
        
    nbr_of_image_pairs = 100
    eval_itr = 5
    
    left,right,sim = generator.prep_eval_match(nbr_of_image_pairs)
        
    output_dir = "/tmp/siamese_finger_model/" # directory where the model is saved
    
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
#            left_full = []
#            right_full = []
            for i in range(eval_itr):
                left,right,sim = generator.prep_eval_match(nbr_of_image_pairs)
                left_o,right_o= sess.run([left_eval_inference,right_eval_inference],feed_dict = {left_eval:left, right_eval:right})
                if i == 0:
                    left_full = left_o
                    right_full = right_o
                    sim_full = sim
                else:
                    left_full = np.vstack((left_full,left_o))
                    right_full = np.vstack((right_full,right_o))
                    sim_full = np.vstack((sim_full, sim))
                
#            left_full = np.array(left_full)
#            right_full = np.array(right_full)

            precision, false_pos, false_neg, recall, fnr, fpr = evaluate_siamese_network(left_full,right_full,sim_full,0.7)

            print("Precision: %f " % precision)
            print("# False positive: %d " % false_pos)
            print("# False negative: %d " % false_neg)
            print("# Recall: %f " % recall)
            print("# Miss rate/false negative rate: %f " % fnr)
            print("# fall-out/false positive rate: %f " % fpr)
      
if __name__ == "__main__":
    tf.app.run()
    
    
    
    
    

        