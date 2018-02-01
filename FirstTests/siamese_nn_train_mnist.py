# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:44:22 2018

@author: Tuong Lam
"""

from data_generator import data_generator
from siamese_nn_model_mnist import *

import numpy as np
import tensorflow as tf
import os 


def main(unused_argv):
    """ This method is used to train a siamese network for the mnist dataset
    
    The model is defined in the file siamese_nn_model_mnist.py. The class
    data_generator is used to generate batches for training. When training
    is completed the model is saved in the file /tmp/siamese_mnist_model/.
    If a model exists it will be used for further training, otherwise a new
    one is created.
    
    """
    
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    output_dir = "/tmp/siamese_mnist_model/" #directory where the model will be saved
    
    generator = data_generator(train_data,train_labels) # initialize data generator
    
    # parameters for training
    batch_size = 50
    train_iter = 2000
    learning_rate = 0.01
    momentum = 0.99
    
    # parameters for evaluation
    nbr_of_pairs = 5000
    
    tf.reset_default_graph()
    
    # if models exists use the existing one otherwise create a new one
    if not os.path.exists(output_dir + ".meta"):
        print("No previous model exists, creating a new one.")
        is_model_new = True

         # create placeholders for pairs of images and ground truth matching
        left,right,label,left_eval,right_eval = placeholder_inputs(batch_size,nbr_of_pairs)
            
        left_output = inference(left)            
        right_output = inference(right)
        left_eval_output = inference(left_eval)            
        right_eval_output = inference(right_eval)
        
        margin = tf.constant(2.0)
        loss = contrastive_loss(left_output,right_output,label,margin)
        tf.add_to_collection("loss",loss)
        tf.add_to_collection("left_output",left_output)
        tf.add_to_collection("right_output",right_output)
        tf.add_to_collection("left_eval_output",left_eval_output)
        tf.add_to_collection("right_eval_output",right_eval_output)
        saver = tf.train.Saver()
        
    else:
        print("Using existing model in the directory " + output_dir)
        is_model_new = False
        
        saver = tf.train.import_meta_graph(output_dir + ".meta")
        g = tf.get_default_graph()
        left = g.get_tensor_by_name("left:0")
        right = g.get_tensor_by_name("right:0")
        label = g.get_tensor_by_name("label:0")
        loss = tf.get_collection("loss")[0]
        left_output = tf.get_collection("left_output")[0]
        right_output = tf.get_collection("right_output")[0]
        
    with tf.Session() as sess:
        if is_model_new:
            train_op = training(loss, learning_rate, momentum)
            sess.run(tf.global_variables_initializer()) # initialize all trainable parameters
            tf.add_to_collection("train_op",train_op)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(output_dir))
            train_op = tf.get_collection("train_op")[0]
#            global_vars = tf.global_variables()
#            for i in range(len(global_vars)):
#                print(global_vars[i])
        
        for i in range(1,train_iter + 1):
            b_l, b_r, b_sim = generator.gen_batch(batch_size)
            _,loss_value,left_o,right_o = sess.run([train_op, loss, left_output, right_output],feed_dict={left:b_l, right:b_r, label:b_sim})
#            print(left_o)
#            print(right_o)
            if i % 100 == 0:
                print("Iteration %d: loss = %.5f" % (i, loss_value))
        
#        graph = tf.get_default_graph()
#        kernel_var = graph.get_tensor_by_name("conv_layer_1/bias:0")
#        kernel_var_after_init = sess.run(kernel_var)
#        dims = np.shape(kernel_var_after_init)
#        print(kernel_var_after_init)
        
        save_path = tf.train.Saver().save(sess,output_dir)
        print("Trained model saved in path: %s" % save_path)
    
    
if __name__ == "__main__":
    tf.app.run()