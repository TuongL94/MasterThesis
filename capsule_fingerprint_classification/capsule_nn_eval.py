#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 12:56:24 2018

@author: Tuong Lam
"""

import numpy as np
import tensorflow as tf
import os 
import sys
import pickle


def evaluate_capsule_network(generator, batch_size, eval_itr, output_dir,gpu_device_name):
    """ This method is used to evaluate a capsule network for fingerprint datasets.
    
    The model is defined in the file capsule_nn_model.py and trained in 
    the file capsule_nn_train.py. Evaluation will only be performed if
    a model exists. The method will print evaluation metrics.
    
    Input:
    generator - an instance of a data_generator object used in training
    batch_size - batch size for the evaluation placeholder
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
                break
        saver = tf.train.import_meta_graph(meta_file_name)
        
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
                
                test_match_dataset = tf.data.Dataset.from_tensor_slices(generator.match_test)
                test_match_dataset = test_match_dataset.batch(batch_size)
#                test_match_dataset_length = np.shape(generator.match_test)[0]
                            
                test_match_iterator = test_match_dataset.make_one_shot_iterator()
                test_match_handle = sess.run(test_match_iterator.string_handle())
                            
                iterator = tf.data.Iterator.from_string_handle(handle, test_match_dataset.output_types)
                next_element = iterator.get_next()
                
                sim_full = np.vstack((np.ones((batch_size*eval_itr,1)), np.zeros((batch_size*eval_itr,1))))
                
                for i in range(eval_itr):
                    test_batch = sess.run(next_element,feed_dict={handle:test_match_handle})
                    for j in range(generator.rotation_res):
                        b_l_test,b_r_test = generator.get_pairs(generator.test_data[j],test_batch)
                        class_id_batch = generator.same_class(test_batch,test=True)
                        left_o,right_o = sess.run([left_test_inference,right_test_inference],feed_dict = {left_test:b_l_test, right_test:b_r_test})
                        if i == 0 and j == 0:
                            left_full = left_o
                            right_full = right_o
                            class_id = class_id_batch
                        else:
                            left_full = np.vstack((left_full,left_o))
                            right_full = np.vstack((right_full,right_o))
                            class_id = np.vstack((class_id, class_id_batch))
                            
                        class_id_batch = generator.same_class(test_batch,test=True)
                        class_id = np.vstack((class_id, class_id_batch))
                        
                nbr_same_class = np.sum(class_id[eval_itr*batch_size:])
                print("Number of fingerprints in the same class in the non matching set: %d " % nbr_same_class)
         
def main(argv):
   """ Runs evaluation on trained network
   """
    
    # Set parameters for evaluation
   batch_size = 4
   eval_itr = 10
    
   output_dir = argv[1] + argv[0] + "/" # directory where the model is saved
   gpu_device_name = argv[-1] 
   
    # Load generator
   with open("generator_data.pk1", "rb") as input:
       generator = pickle.load(input)
    
   evaluate_capsule_network(generator, batch_size, eval_itr, output_dir, gpu_device_name)
    
if __name__ == "__main__":
    main(sys.argv[1:])