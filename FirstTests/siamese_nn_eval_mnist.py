# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:34:36 2018

@author: Tuong Lam
"""

from siamese_nn_model_mnist import *

import numpy as np
import scipy.linalg as sl
import tensorflow as tf
import os 

def prep_eval_data(eval_data,eval_labels):
    nbr_of_images,dim_squarred = np.shape(eval_data)
    dim = int(np.sqrt(dim_squarred))
    nbr_of_image_pairs = int(nbr_of_images/2)
    eval_data_moded = eval_data.reshape((nbr_of_images,dim,dim,1))
    left = []
    right = []
    sim = np.zeros((nbr_of_image_pairs,1))
    for i in range(nbr_of_image_pairs):
        left.append(eval_data_moded[i,:,:,:])
        right.append(eval_data_moded[nbr_of_image_pairs+i,:,:,:])
        if(eval_labels[i] == eval_labels[nbr_of_image_pairs + i]):
            sim[i] = 1
            
    return np.array(left),np.array(right),sim
    
def evaluate_mnist_siamese_network(left_pairs,right_pairs,sim_labels,threshold):
    left_pairs_inference = inference(tf.convert_to_tensor(left_pairs))
    right_pairs_inference = inference(tf.convert_to_tensor(right_pairs))
    matching = np.zeros(len(sim_labels))
    
    for i in range(len(sim_labels)):
        if tf.linalg.norm([left_pairs_inference[i,:],right_pairs_inference[i,:]]).eval() < threshold:
            matching[i] = 1
    
    precision = np.sum((matching == sim_labels))/len(sim_labels)
    return precision
    
    
    
def main(unused_argv):
    """ This method is used to evaluate a siamese network for the mnist dataset
    
    The model is defined in the file siamese_nn_model_mnist.py and trained in 
    the file siamese_nn_train_mnist.py. Evaluation will only be performed if
    a model exists.
    
    """
    
    # Load eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    left,right,sim = prep_eval_data(eval_data,eval_labels)
    
    output_dir = "/tmp/siamese_mnist_model/" #directory where the model is saved
    
    tf.reset_default_graph()
    
    if not os.path.exists(output_dir + ".meta"):
        print("No siamese model exists in " + output_dir)
        return
        
    else:
        print("Using existing model in the directory " + output_dir + " for evaluation")
        
        saver = tf.train.import_meta_graph(output_dir + ".meta")
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(output_dir))
            precision = sess.run([evaluate_mnist_siamese_network(left,right,sim,3)])
            print("Precision of siamese network: " + precision)
      
if __name__ == "__main__":
    tf.app.run()
    
    
    
    
    

        