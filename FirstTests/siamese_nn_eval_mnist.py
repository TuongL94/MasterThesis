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


def evaluate_mnist_siamese_network(left_pairs_o,right_pairs_o,sim_labels,threshold):
    matching = np.zeros(len(sim_labels))
    l2_normalized_diff = util.l2_normalize(left_pairs_o-right_pairs_o)
    false_pos = 0
    false_neg = 0
    p = np.sum(sim_labels)
    n = len(sim_labels) - p
    for i in range(len(sim_labels)):
        if sl.norm(l2_normalized_diff[i,:]) < threshold:
            matching[i] = 1
            if sim_labels[i] == [0]:
                false_pos = false_pos + 1
        else:
            if sim_labels[i] == [1]:
                false_neg = false_neg + 1
    
    precision = np.sum((matching == sim_labels))/len(sim_labels)
    tp = 0
    for i in range(len(sim_labels)):
        if matching[i] == 1 and sim_labels[i] == 1:
            tp += 1
    recall = tp/p
    fnr = 1 - recall
    fpr = false_pos/n
    
    return precision, false_pos, false_neg, recall, fnr, fpr

def main(unused_argv):
    """ This method is used to evaluate a siamese network for the mnist dataset.
    
    The model is defined in the file siamese_nn_model_mnist.py and trained in 
    the file siamese_nn_train_mnist.py. Evaluation will only be performed if
    a model exists.
    
    """
    # Load mnist eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    eval_data = util.reshape_grayscale_data(mnist.test.images) # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    left,right,sim = util.prep_eval_data_pair(eval_data,eval_labels)
    
#    left,right,sim = util.prep_eval_data(eval_data,eval_labels)
    
    output_dir = "/tmp/siamese_mnist_model/" # directory where the model is saved
    
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
            left_o,right_o= sess.run([left_eval_inference,right_eval_inference],feed_dict = {left_eval:left, right_eval:right})

            precision, false_pos, false_neg, recall, fnr, fpr = evaluate_mnist_siamese_network(left_o,right_o,sim,0.5)
            print("Precision: %f " % precision)
            print("# False positive: %d " % false_pos)
            print("# False negative: %d " % false_neg)
            print("# Recall: %f " % recall)
            print("# Miss rate/false negative rate: %f " % fnr)
            print("# fall-out/false positive rate: %f " % fpr)
      
if __name__ == "__main__":
    tf.app.run()
    
    
    
    
    

        