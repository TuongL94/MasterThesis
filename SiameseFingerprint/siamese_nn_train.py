# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:44:22 2018

@author: Tuong Lam & Simon Nilsson
"""

from data_generator import data_generator

import siamese_nn_model as sm
import numpy as np
import tensorflow as tf
import os 
import utilities as util
import siamese_nn_eval as sme
import matplotlib.pyplot as plt
import pickle
import re
import sys


def main(argv):
    """ This method is used to train a siamese network for fingerprint datasets.
    
    The model is defined in the file siamese_nn_model.py.
    When training is completed the model is saved in the file /tmp/siamese_finger_model/.
    If a model exists it will be used for further training, otherwise a new
    one is created. It is also possible to evaluate the model directly after training.
    
    """
    
    gpu_device_name = argv
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir = dir_path + "/train_models" + gpu_device_name + "/" # directory where the model will be saved
    
    # Load fingerprint data and create a data_generator instance if one 
    # does not exist, otherwise load existing data_generator
    if not os.path.exists(dir_path + "/generator_data.pk1"):
        with open('generator_data.pk1', 'wb') as output:
            # Load fingerprint labels and data from file with names
            finger_id = np.load(dir_path + "/finger_id.npy")
            person_id = np.load(dir_path + "/person_id.npy")
            finger_data = np.load(dir_path + "/fingerprints.npy")
            translation = np.load(dir_path + "/translation.npy")
            rotation = np.load(dir_path + "/rotation.npy")
#            nbr_of_images = np.shape(finger_data)[0] # number of images to use from the original data set
            nbr_of_images = 5616
            finger_data = util.reshape_grayscale_data(finger_data)
            rotation_res = 1
            generator = data_generator(finger_data, finger_id, person_id, translation, rotation, nbr_of_images, rotation_res) # initialize data generator
            pickle.dump(generator, output, pickle.HIGHEST_PROTOCOL)
    else:
        # Load generator
        with open('generator_data.pk1', 'rb') as input:
            generator = pickle.load(input)
    
    # parameters for training
    batch_size_train = 150
    train_iter = 30000
    learning_rate = 0.00001
    momentum = 0.99
   
    # parameters for validation
    batch_size_val = 175
    
    # parameters for evaluation
    batch_size_test = 105
    threshold = 0.5    
        
    dims = np.shape(generator.train_data[0])
    batch_sizes = [batch_size_train,batch_size_val,batch_size_test]
    image_dims = [dims[1],dims[2],dims[3]]
    
    save_itr = 10000 # frequency in which the model is saved
    
    tf.reset_default_graph()
    
    # if models exists use the latest existing one otherwise create a new one
    if not os.path.exists(output_dir + "checkpoint"):
        print("No previous model exists, creating a new one.")
        is_model_new = True
        current_itr = 0 # current training iteration
        
        with tf.device(gpu_device_name):
             # create placeholders
            left_train,right_train,label_train,left_val,right_val,label_val,left_test,right_test = sm.placeholder_inputs(image_dims,batch_sizes)
            handle = tf.placeholder(tf.string, shape=[],name="handle")
                
            left_train_output = sm.inference(left_train)            
            right_train_output = sm.inference(right_train)
            left_val_output = sm.inference(left_val)
            right_val_output = sm.inference(right_val)
            left_test_output = sm.inference(left_test)
            right_test_output = sm.inference(right_test)
            
            margin = tf.constant(4.0) # margin for contrastive loss
            train_loss = sm.contrastive_loss(left_train_output,right_train_output,label_train,margin)
            
            val_loss = sm.contrastive_loss(left_val_output,right_val_output,label_val,margin)
            
            tf.add_to_collection("train_loss",train_loss)
            tf.add_to_collection("val_loss",val_loss)
            tf.add_to_collection("left_val_output",left_val_output)
            tf.add_to_collection("right_val_output",right_val_output)
            tf.add_to_collection("left_test_output",left_test_output)
            tf.add_to_collection("right_test_output",right_test_output)
            
            saver = tf.train.Saver()

    else:
        print("Using latest existing model in the directory " + output_dir)
        is_model_new = False
        
        with open(output_dir + "checkpoint","r") as file:
            line  = file.readline()
            words = re.split("/",line)
            model_file_name = words[-1][:-2]
            current_itr = int(re.split("-",model_file_name)[-1]) # current training iteration
            saver = tf.train.import_meta_graph(output_dir + model_file_name + ".meta")
        
        with tf.device(gpu_device_name):
            g = tf.get_default_graph()
            left_train = g.get_tensor_by_name("left_train:0")
            right_train = g.get_tensor_by_name("right_train:0")
            label_train = g.get_tensor_by_name("label_train:0")
            train_loss = tf.get_collection("train_loss")[0]
            left_val_output = tf.get_collection("left_val_output")[0]
            right_val_output = tf.get_collection("right_val_output")[0]
            
            left_val = g.get_tensor_by_name("left_val:0")
            right_val = g.get_tensor_by_name("right_val:0")
            label_val = g.get_tensor_by_name("label_val:0")
            val_loss = tf.get_collection("val_loss")[0]
            
            handle= g.get_tensor_by_name("handle:0")
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if is_model_new:
            with tf.device(gpu_device_name):
                train_op = sm.training(train_loss, learning_rate, momentum)
                sess.run(tf.global_variables_initializer()) # initialize all trainable parameters
                tf.add_to_collection("train_op",train_op)
        else:
            with tf.device(gpu_device_name):
                saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                train_op = tf.get_collection("train_op")[0]
            
#            for i in sess.graph.get_operations():
#                print(i.values())
#            global_vars = tf.global_variables()
#            for i in range(len(global_vars)):
#                print(global_vars[i])
        with tf.device(gpu_device_name): 
            graph = tf.get_default_graph()
            conv1_layer = graph.get_tensor_by_name("conv_layer_1/kernel:0")
            nbr_of_filters_conv1 = sess.run(tf.shape(conv1_layer)[-1])
    
            conv2_layer = graph.get_tensor_by_name("conv_layer_2/kernel:0")
            hist_conv1 = tf.summary.histogram("hist_conv1", conv1_layer)
            hist_conv2 = tf.summary.histogram("hist_conv2", conv2_layer)
            conv1_layer = tf.transpose(conv1_layer, perm = [3,0,1,2])
            filter1 = tf.summary.image('Filter_1', conv1_layer, max_outputs=nbr_of_filters_conv1)
    #        conv2_layer = tf.transpose(conv2_layer, perm = [3,0,1,2])
    #        filter2 = tf.summary.image('Filter_2', conv2_layer, max_outputs=32)
            bias_conv1 = graph.get_tensor_by_name("conv_layer_1/bias:0")
            hist_bias1 = tf.summary.histogram("hist_bias1", bias_conv1)
            bias_conv2 = graph.get_tensor_by_name("conv_layer_2/bias:0")
            hist_bias2 = tf.summary.histogram("hist_bias2", bias_conv2)
    
                
            summary_op = tf.summary.scalar('training_loss', train_loss)
    #        summary_val_loss = tf.summary.scalar("validation_loss",val_loss)
            x_image = tf.summary.image('input', left_train)
            summary_op = tf.summary.merge([summary_op, x_image, filter1, hist_conv1, hist_conv2, hist_bias1, hist_bias2])
            # Summary setup
            writer = tf.summary.FileWriter(output_dir + "/summary", graph=tf.get_default_graph())
          
#        counter = 0
#        data_size = batch_size
        precision_over_time = []
        thresh_step = 0.05
        
        with tf.device(gpu_device_name):
            # Setup tensorflow's batch generator
            train_match_dataset = tf.data.Dataset.from_tensor_slices(generator.match_train)
    #        train_match_dataset = train_match_dataset.map(lambda x: x**2)
            train_match_dataset = train_match_dataset.shuffle(buffer_size=np.shape(generator.match_train)[0])
            train_match_dataset = train_match_dataset.repeat()
            train_match_dataset = train_match_dataset.batch(int(batch_size_train/2))
            
            train_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.no_match_train)
            train_non_match_dataset = train_non_match_dataset.shuffle(buffer_size=np.shape(generator.no_match_train)[0])
            train_non_match_dataset = train_non_match_dataset.repeat()
            train_non_match_dataset = train_non_match_dataset.batch(int((batch_size_train+1)/2))
            
            val_match_dataset_length = np.shape(generator.match_val)[0]
            val_match_dataset = tf.data.Dataset.from_tensor_slices(generator.match_val)
            val_match_dataset = val_match_dataset.shuffle(buffer_size = val_match_dataset_length)
            val_match_dataset = val_match_dataset.repeat()
            val_match_dataset = val_match_dataset.batch(batch_size_val)
            
            val_non_match_dataset_length = np.shape(generator.no_match_val)[0]
            val_non_match_dataset = tf.data.Dataset.from_tensor_slices(generator.no_match_val[0:int(val_non_match_dataset_length/10)])
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
        
        # Training loop
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
            
            # Randomize rotation of batch              
            rnd_rotation = np.random.randint(0,generator.rotation_res)
            b_l_train,b_r_train = generator.get_pairs(generator.train_data[rnd_rotation],train_batch)
            
            _,train_loss_value, summary = sess.run([train_op, train_loss, summary_op],feed_dict={left_train:b_l_train, right_train:b_r_train, label_train:b_sim_train})

             # Use validation data set to tune hyperparameters (Classification threshold)
            if i % 1000 == 0:
                b_sim_val_matching = np.ones((batch_size_val*int(val_match_dataset_length/batch_size_val),1))
                b_sim_val_non_matching = np.zeros((batch_size_val*int(val_match_dataset_length/batch_size_val),1))
#                b_sim_val_non_matching = np.zeros((batch_size_val*int((int(val_non_match_dataset_length/10)+1)/batch_size_val),1))
                b_sim_val = np.append(b_sim_val_matching,b_sim_val_non_matching,axis=0)
                for j in range(int(val_match_dataset_length/batch_size_val)):
                    val_batch_matching = sess.run(next_element,feed_dict={handle:val_match_handle})
                    for k in range(generator.rotation_res):
                        b_l_val,b_r_val = generator.get_pairs(generator.val_data[k],val_batch_matching) 
                        left_o,right_o = sess.run([left_val_output,right_val_output],feed_dict = {left_val:b_l_val, right_val:b_r_val})
                        if j == 0 and k == 0:
                            left_full = left_o
                            right_full = right_o
                        else:
                            left_full = np.vstack((left_full,left_o))
                            right_full = np.vstack((right_full,right_o))
                    
#                for k in range(int((int(val_non_match_dataset_length/10)+1)/batch_size_val)):
                for j in range(int(val_match_dataset_length/batch_size_val)):
                    val_batch_non_matching = sess.run(next_element,feed_dict={handle:val_non_match_handle})
#                    for l in range(generator.rotation_res):
                    b_l_val,b_r_val = generator.get_pairs(generator.val_data[0],val_batch_non_matching) 
                    left_o,right_o = sess.run([left_val_output,right_val_output],feed_dict = {left_val:b_l_val, right_val:b_r_val})
                    left_full = np.vstack((left_full,left_o))
                    right_full = np.vstack((right_full,right_o)) 
                
                precision, false_pos, false_neg, recall, fnr, fpr = sme.get_test_diagnostics(left_full,right_full, b_sim_val,threshold)
            
                if false_pos > false_neg:
                    threshold -= thresh_step
                else:
                    threshold += thresh_step
                precision_over_time.append(precision)
            
#            if i % 100 == 0:
#                print("Iteration %d: train loss = %.5f" % (i, train_loss_value))
#                print("Iteration %d: val loss = %.5f" % (i,val_loss_value))
            writer.add_summary(summary, i)
            
            if i % save_itr == 0:
                save_path = tf.train.Saver().save(sess,output_dir + "model",global_step=i+current_itr)
                print("Trained model after {} iterations saved in path: {}".format(i,save_path))
        
        # Plot precision over time
        time = list(range(len(precision_over_time)))
        plt.plot(time, precision_over_time)
        plt.show()
        
        print("Current threshold: %f" % threshold)
        print("Final precision: %f" % precision_over_time[-1])
        
if __name__ == "__main__":
#    tf.app.run()
     main(sys.argv[1])