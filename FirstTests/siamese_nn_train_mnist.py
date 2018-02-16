# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:44:22 2018

@author: Tuong Lam
"""

from data_generator import data_generator

import siamese_nn_model_mnist as sm
import numpy as np
import tensorflow as tf
import os 
import utilities as util
import siamese_nn_eval_mnist as sme
import matplotlib.pyplot as plt
import pickle


def main(unused_argv):
    """ This method is used to train a siamese network for the mnist dataset.
    
    The model is defined in the file siamese_nn_model_mnist.py. The class
    data_generator is used to generate batches for training. When training
    is completed the model is saved in the file /tmp/siamese_mnist_model/.
    If a model exists it will be used for further training, otherwise a new
    one is created.
    
    """
        
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir = "/tmp/siamese_mnist_model/" # directory where the model will be saved
        
    # Load mnist data and create a data_generator instance if one 
    # does not exist, otherwise load existing data_generator
    if not os.path.exists(dir_path + "/generator_data.pk1"):
        with open('generator_data.pk1', 'wb') as output:
            # Load mnist training and eval data and perform necessary data reshape
            mnist = tf.contrib.learn.datasets.load_dataset("mnist")
            train_data = util.reshape_grayscale_data(mnist.train.images) # Returns np.array
            train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#            nbr_of_images = np.shape(train_data)[0] # number of images to use from the original data set
            nbr_of_images = 10000
            generator = data_generator(train_data,train_labels,nbr_of_images) # initialize data generator
            pickle.dump(generator, output, pickle.HIGHEST_PROTOCOL)
    else:
        # Load generator
        with open('generator_data.pk1', 'rb') as input:
            generator = pickle.load(input)
    
    # parameters for training
    batch_size = 1000
    train_iter = 2000
    learning_rate = 0.001
    momentum = 0.99
    
    image_dims = np.shape(generator.images)
    placeholder_dims = [batch_size, image_dims[1], image_dims[2], image_dims[3]] 
    
    # parameters for validation
    batch_size_val = 100

    # parameters for evaluation
    nbr_of_eval_pairs = 100
    eval_itr = 10
    threshold = 0.15 
    
    tf.reset_default_graph()
    
    # if models exists use the existing one otherwise create a new one
    if not os.path.exists(output_dir + ".meta"):
        print("No previous model exists, creating a new one.")
        is_model_new = True

         # create placeholders
#        left,right,label,left_eval,right_eval = sm.placeholder_inputs(placeholder_dims,nbr_of_eval_pairs)
            
        left,right,label,left_val,right_val,label_val,left_eval,right_eval = sm.placeholder_inputs(placeholder_dims,nbr_of_eval_pairs,batch_size_val)
        
        left_output = sm.inference(left)            
        right_output = sm.inference(right)
        left_eval_output = sm.inference(left_eval)            
        right_eval_output = sm.inference(right_eval)
        left_val_output = sm.inference(left_val)
        right_val_output = sm.inference(right_val)
        
        margin = tf.constant(2.0)
        train_loss = sm.contrastive_loss(left_output,right_output,label,margin)
        
        val_loss = sm.contrastive_loss(left_val_output,right_val_output,label_val,margin)
        
        tf.add_to_collection("train_loss",train_loss)
        tf.add_to_collection("val_loss",val_loss)
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
        train_loss = tf.get_collection("train_loss")[0]
        left_output = tf.get_collection("left_output")[0]
        right_output = tf.get_collection("right_output")[0]
        
        left_val = g.get_tensor_by_name("left_val:0")
        right_val = g.get_tensor_by_name("right_val:0")
        label_val = g.get_tensor_by_name("label_val:0")
        val_loss = tf.get_collection("val_loss")[0]
        
    with tf.Session() as sess:
        if is_model_new:
            train_op = sm.training(train_loss, learning_rate, momentum)
            sess.run(tf.global_variables_initializer()) # initialize all trainable parameters
            tf.add_to_collection("train_op",train_op)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(output_dir))
            train_op = tf.get_collection("train_op")[0]
            
#            for i in sess.graph.get_operations():
#                print(i.values())
#            global_vars = tf.global_variables()
#            for i in range(len(global_vars)):
#                print(global_vars[i])
        graph = tf.get_default_graph()
        conv1_layer = graph.get_tensor_by_name("conv_layer_1/kernel:0")
        nbr_of_filters_conv1 = sess.run(tf.shape(conv1_layer)[-1])

        conv2_layer = graph.get_tensor_by_name("conv_layer_2/kernel:0")
        hist_conv1 = tf.summary.histogram("hist_conv1", conv1_layer)
        hist_conv2 = tf.summary.histogram("hist_conv2", conv2_layer)
        conv1_layer = tf.transpose(conv1_layer, perm = [3,0,1,2])
        filter1 = tf.summary.image('Filter_1', conv1_layer, max_outputs=nbr_of_filters_conv1)
        conv1_layer = tf.transpose(conv1_layer, perm = [1,2,3,0])
#        conv2_layer = tf.transpose(conv2_layer, perm = [3,0,1,2])
#        filter2 = tf.summary.image('Filter_2', conv2_layer, max_outputs=32)
        bias_conv1 = graph.get_tensor_by_name("conv_layer_1/bias:0")
        hist_bias1 = tf.summary.histogram("hist_bias1", bias_conv1)
        bias_conv2 = graph.get_tensor_by_name("conv_layer_2/bias:0")
        hist_bias2 = tf.summary.histogram("hist_bias2", bias_conv2)
        
        summary_op = tf.summary.scalar('training_loss', train_loss)
        summary_val_loss = tf.summary.scalar("validation_loss",val_loss)
        x_image = tf.summary.image('input', left)
        summary_op = tf.summary.merge([summary_op, x_image, filter1, hist_conv1, hist_conv2, hist_bias1, hist_bias2,summary_val_loss])
        # Summary setup
        writer = tf.summary.FileWriter(output_dir + "/summary", graph=tf.get_default_graph())
        
        precision_over_time = []
        thresh_step = 0.05
        
        for i in range(1,train_iter + 1):
            b_l, b_r, b_sim = generator.gen_batch(batch_size)
            b_val_l, b_val_r, b_val_sim = generator.gen_batch(batch_size_val,training = 0)
            _,train_loss_value, val_loss_value,left_o,right_o, summary = sess.run([train_op, train_loss, val_loss, left_output, right_output, summary_op],feed_dict={left:b_l, right:b_r, label:b_sim, left_val:b_val_l, right_val:b_val_r,label_val:b_val_sim})
#            print(left_o)
#            print(right_o)
            if i % 100 == 0:
                print("Iteration %d: training loss = %.5f" % (i, train_loss_value))
                print("Iteration %d: validation loss = %.5f" % (i,val_loss_value))
                
            writer.add_summary(summary, i)
            precision, false_pos, false_neg, recall, fnr, fpr = sme.get_eval_diagnostics(left_o,right_o, b_sim,threshold)
            
            if false_pos > false_neg:
                threshold -= thresh_step
            else:
                threshold += thresh_step   
            precision_over_time.append(precision)
#        graph = tf.get_default_graph()
#        kernel_var = graph.get_tensor_by_name("conv_layer_1/bias:0")
#        kernel_var_after_init = sess.run(kernel_var)
#        dims = np.shape(kernel_var_after_init)
#        print(kernel_var_after_init)
        
        save_path = tf.train.Saver().save(sess,output_dir)
        print("Trained model saved in path: %s" % save_path)
        
        # Plot precision over time
        time = list(range(len(precision_over_time)))
        plt.plot(time, precision_over_time)
        plt.show()
        print("Current threshold: %f" % threshold)
        
    # Only run this if the final network is to be evaluated    
    sme.evaluate_siamese_network(generator,nbr_of_eval_pairs,eval_itr,threshold,output_dir)
    
if __name__ == "__main__":
    tf.app.run()