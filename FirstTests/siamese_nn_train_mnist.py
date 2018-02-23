# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:44:22 2018

@author: Tuong Lam & Simon Nilsson
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
    
    The network uses kernels and biases from the first convolutional layer of a
    trained classification network for mnist. Thus one has to first train such
    a network to be able to train this network.
    The model is defined in the file siamese_nn_model_mnist.py. When training
    is completed the model is saved in the file /tmp/siamese_mnist_model/.
    If a model exists it will be used for further training, otherwise a new
    one is created.
    
    """
        
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir = "/tmp/siamese_mnist_model/" # directory where the model will be saved
    mnist_class_dir = "/tmp/single_mnist_model/" # directory where the classification network is saved
        
    # Load mnist data and create a data_generator instance if one 
    # does not exist, otherwise load existing data_generator
    if not os.path.exists(dir_path + "/generator_data.pk1"):
        with open('generator_data.pk1', 'wb') as output:
            mnist = tf.contrib.learn.datasets.load_dataset("mnist")
            train_data = util.reshape_grayscale_data(mnist.train.images)
            train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
            val_data = util.reshape_grayscale_data(mnist.validation.images)
            val_labels = np.asarray(mnist.validation.labels, dtype=np.int32)
            test_data = util.reshape_grayscale_data(mnist.test.images)
            test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
            data_sizes = [1000,1000,1000] # number of samples to use from each data set
            generator = data_generator(train_data,train_labels,val_data,val_labels,test_data,test_labels,data_sizes) # initialize data generator
            pickle.dump(generator, output, pickle.HIGHEST_PROTOCOL)
    else:
        # Load generator
        with open('generator_data.pk1', 'rb') as input:
            generator = pickle.load(input)
    
    # parameters for training
    batch_size_train = 1000
    train_iter = 1000
    learning_rate = 0.001
    momentum = 0.99
        
    # parameters for validation
    batch_size_val = 1000

    # parameters for testing
    batch_size_test = 1000
    threshold = 0.5
    
    dims = np.shape(generator.train_data)
    batch_sizes = [batch_size_train,batch_size_val,batch_size_test]
    image_dims = [dims[1],dims[2],dims[3]] 
    
    tf.reset_default_graph()
    
    # if models exists use the existing one otherwise create a new one
    if not os.path.exists(output_dir + ".meta"):
        print("No previous model exists, creating a new one with weights from siamese classification network.")
        is_model_new = True
        old_saver = tf.train.import_meta_graph(mnist_class_dir + ".meta")
        g = tf.get_default_graph()
        start_kernel_layer_1 = None
        start_bias_layer_1 = None
        with tf.Session() as sess:
            old_saver.restore(sess, tf.train.latest_checkpoint(mnist_class_dir))
            old_kernel= g.get_tensor_by_name("inference/conv1/weight:0")
            start_kernel_layer_1 = sess.run(old_kernel)
            old_bias = g.get_tensor_by_name("inference/conv1/bias:0")
            start_bias_layer_1 = sess.run(old_bias)
                        
        graph = tf.Graph()
        with graph.as_default():
             # create placeholders            
            left_train,right_train,label_train,left_val,right_val,label_val,left_test,right_test = sm.placeholder_inputs(image_dims,batch_sizes)  
            handle = tf.placeholder(tf.string, shape=[],name="handle")

            transfer = (start_kernel_layer_1,start_bias_layer_1)
            with tf.variable_scope("inference",reuse=tf.AUTO_REUSE):
                left_train_output = sm.inference(left_train,transfer)
                right_train_output = sm.inference(right_train,transfer)
                left_val_output = sm.inference(left_val,transfer)
                right_val_output = sm.inference(right_val,transfer)
                left_test_output = sm.inference(left_test,transfer)            
                right_test_output = sm.inference(right_test,transfer)
                            
            margin = tf.constant(4.0)
            train_loss = sm.contrastive_loss(left_train_output,right_train_output,label_train,margin)
            val_loss = sm.contrastive_loss(left_val_output,right_val_output,label_val,margin)
            
            tf.add_to_collection("train_loss",train_loss)
            tf.add_to_collection("val_loss",val_loss)
            tf.add_to_collection("left_val_output",left_val_output)
            tf.add_to_collection("right_val_output",right_val_output)
            tf.add_to_collection("left_test_output",left_test_output)
            tf.add_to_collection("right_test_output",right_test_output)
    else:
        print("Using existing model in the directory " + output_dir)
        is_model_new = False
        
        saver = tf.train.import_meta_graph(output_dir + ".meta")
        g = tf.get_default_graph()
        
        left_train = g.get_tensor_by_name("left_train:0")
        right_train = g.get_tensor_by_name("right_train:0")
        label_train = g.get_tensor_by_name("label_train:0")
        train_loss = tf.get_collection("train_loss")[0]
        
        left_val = g.get_tensor_by_name("left_val:0")
        right_val = g.get_tensor_by_name("right_val:0")
        label_val = g.get_tensor_by_name("label_val:0")
        val_loss = tf.get_collection("val_loss")[0]
        left_val_output = tf.get_collection("left_val_output")[0]
        right_val_output = tf.get_collection("right_val_output")[0]
        
        handle= g.get_tensor_by_name("handle:0")
        
    with tf.Session(graph = graph) as sess:
        if is_model_new:
            train_op = sm.training(train_loss,learning_rate,momentum)
            sess.run(tf.global_variables_initializer())
            tf.add_to_collection("train_op",train_op)
            
        else:
            saver.restore(sess, tf.train.latest_checkpoint(output_dir))
            train_op = tf.get_collection("train_op")[0]

        conv1_weight = graph.get_tensor_by_name("inference/conv1/weight:0")
        nbr_of_filters_conv1 = sess.run(tf.shape(conv1_weight)[-1])
        hist_conv1_weight = tf.summary.histogram("hist_conv1_weight", conv1_weight)
        conv1_weight = tf.transpose(conv1_weight, perm = [3,0,1,2])
        filter1 = tf.summary.image('Filter_1', conv1_weight, max_outputs=nbr_of_filters_conv1)

        conv1_bias = graph.get_tensor_by_name("inference/conv1/bias:0")
        hist_conv1_bias = tf.summary.histogram("hist_conv1_bias", conv1_bias)
        
        conv2_weight = graph.get_tensor_by_name("inference/conv2/weight:0")
        hist_conv2_weight = tf.summary.histogram("hist_conv2_weight", conv2_weight)
        
        conv2_bias = graph.get_tensor_by_name("inference/conv2/bias:0")
        hist_conv2_bias = tf.summary.histogram("hist_conv2_bias", conv2_bias)

        summary_train_loss = tf.summary.scalar('training_loss', train_loss)
        summary_val_loss = tf.summary.scalar("validation_loss",val_loss)

        summary_op = tf.summary.merge([summary_train_loss,filter1, hist_conv1_weight, hist_conv1_bias,hist_conv2_weight,hist_conv2_bias])
        
        # Summary setup
        writer = tf.summary.FileWriter(output_dir + "/summary", graph=tf.get_default_graph())
        
        precision_over_time = []
        thresh_step = 0.005
        
        train_match_dataset = tf.data.Dataset.from_tensor_slices(generator.all_match_train)
        train_match_dataset = train_match_dataset.shuffle(buffer_size=np.shape(generator.all_match_train)[0])
        train_match_dataset = train_match_dataset.repeat()
        train_match_dataset = train_match_dataset.batch(int(batch_size_train/2))
        
        train_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.all_non_match_train)
        train_non_match_dataset = train_non_match_dataset.shuffle(buffer_size=np.shape(generator.all_non_match_train)[0])
        train_non_match_dataset = train_non_match_dataset.repeat()
        train_non_match_dataset = train_non_match_dataset.batch(int((batch_size_train+1)/2))
        
        val_match_dataset_length = np.shape(generator.all_match_val)[0]
        val_match_dataset = tf.data.Dataset.from_tensor_slices(generator.all_match_val)
        val_match_dataset = val_match_dataset.shuffle(buffer_size = val_match_dataset_length)
        val_match_dataset = val_match_dataset.repeat()
        val_match_dataset = val_match_dataset.batch(batch_size_val)
        
        val_non_match_dataset_length = np.shape(generator.all_non_match_val)[0]
        val_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.all_non_match_val[0:int(val_non_match_dataset_length/10)])
        val_non_match_dataset = val_non_match_dataset.shuffle(buffer_size = val_non_match_dataset_length)
        val_non_match_dataset = val_non_match_dataset.repeat()
        val_non_match_dataset = val_non_match_dataset.batch(batch_size_val)
        
        train_match_iterator = train_match_dataset.make_one_shot_iterator()
        train_match_handle = sess.run(train_match_iterator.string_handle())
        
        val_match_iterator = val_match_dataset.make_one_shot_iterator()
        val_match_handle = sess.run(val_match_iterator.string_handle())
        
        train_non_match_iterator = train_non_match_dataset.make_one_shot_iterator()
        train_non_match_handle = sess.run(train_non_match_iterator.string_handle())
        
        val_non_match_iterator = val_non_match_dataset.make_one_shot_iterator()
        val_non_match_handle = sess.run(val_non_match_iterator.string_handle())
        
        iterator = tf.data.Iterator.from_string_handle(handle, train_match_dataset.output_types)
        next_element = iterator.get_next()
        
        for i in range(1,train_iter + 1):
                
            train_batch_matching = sess.run(next_element,feed_dict={handle:train_match_handle})
            b_sim_train_matching = np.ones((np.shape(train_batch_matching)[0],1),dtype=np.int32)
            train_batch_non_matching = sess.run(next_element,feed_dict={handle:train_non_match_handle})
            b_sim_train_non_matching = np.zeros((np.shape(train_batch_non_matching)[0],1),dtype=np.int32)
            
            train_batch = np.append(train_batch_matching,train_batch_non_matching,axis=0)
            b_sim_train = np.append(b_sim_train_matching,b_sim_train_non_matching,axis=0)
            permutation = np.random.permutation(batch_size_train)
            train_batch = np.take(train_batch,permutation,axis=0)
            b_sim_train = np.take(b_sim_train,permutation,axis=0)
            
            b_l_train,b_r_train = generator.get_pairs(generator.train_data,train_batch)
            _,train_loss_value,summary = sess.run([train_op, train_loss, summary_op],feed_dict={left_train:b_l_train, right_train:b_r_train, label_train:b_sim_train})
            
            if i % 100 == 0:
                b_sim_val_matching = np.ones((batch_size_val*int(val_match_dataset_length/batch_size_val),1))
                b_sim_val_non_matching = np.zeros((batch_size_val*int((int(val_non_match_dataset_length/10)+1)/batch_size_val),1))
                b_sim_val = np.append(b_sim_val_matching,b_sim_val_non_matching,axis=0)
                for j in range(int(val_match_dataset_length/batch_size_val)):
                    val_batch_matching = sess.run(next_element,feed_dict={handle:val_match_handle})
                    b_l_val,b_r_val = generator.get_pairs(generator.val_data,val_batch_matching) 
                    left_o,right_o = sess.run([left_val_output,right_val_output],feed_dict = {left_val:b_l_val, right_val:b_r_val})
                    if j == 0:
                        left_full = left_o
                        right_full = right_o
                    else:
                        left_full = np.vstack((left_full,left_o))
                        right_full = np.vstack((right_full,right_o))
                    
                for k in range(int((int(val_non_match_dataset_length/10)+1)/batch_size_val)):
                    val_batch_non_matching = sess.run(next_element,feed_dict={handle:val_non_match_handle})
                    b_l_val,b_r_val = generator.get_pairs(generator.val_data,val_batch_non_matching) 
                    left_o,right_o = sess.run([left_val_output,right_val_output],feed_dict = {left_val:b_l_val, right_val:b_r_val})
                    left_full = np.vstack((left_full,left_o))
                    right_full = np.vstack((right_full,right_o)) 
                
                precision, false_pos, false_neg, recall, fnr, fpr = sme.get_test_diagnostics(left_full,right_full, b_sim_val,threshold)
            
                if false_pos > false_neg:
                    threshold -= thresh_step
                else:
                    threshold += thresh_step   
                precision_over_time.append(precision)
                    
            if i % 100 == 0:
                print("Iteration %d: loss = %.5f" % (i, train_loss_value))
#                    print("Iteration %d: val loss = %.5f" % (i,val_loss_value))
                
            writer.add_summary(summary, i)
                        
        save_path = tf.train.Saver().save(sess,output_dir)
        print("Trained model saved in path: %s" % save_path)
        
        # Plot precision over time
        time = list(range(len(precision_over_time)))
        plt.plot(time, precision_over_time)
        plt.show()
        print("Current threshold: %f" % threshold)
            
if __name__ == "__main__":
    tf.app.run()