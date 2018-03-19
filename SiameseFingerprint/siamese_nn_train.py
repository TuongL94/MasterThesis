# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:44:22 2018

@author: Tuong Lam & Simon Nilsson
"""

import numpy as np
import tensorflow as tf
import os 
import matplotlib.pyplot as plt
import pickle
import re
import sys
import time

# imports from self-implemented modules
import siamese_nn_model as sm
import siamese_nn_eval as sme
import siamese_nn_utilities as su
import utilities as util
from data_generator import data_generator

def main(argv):
    """ This method is used to train a siamese network for fingerprint datasets.
    
    The model is defined in the file siamese_nn_model.py.
    If a model exists it will be used for further training, otherwise a new
    one is created. It is also possible to evaluate the model directly after training.
    
    Input:
    argv - arguments to run this method
    argv[0] - path of the directory which the model will be saved in
    argv[1] - name of the GPU to use for training
    argv[2] - optional argument, if this argument is given the model will train for argv[2] minutes
              otherwise it will train for a given amount of iterations
    """
    
    output_dir = argv[0]
    gpu_device_name = argv[1]
    if len(argv) == 3:
        use_time = True
    else:
        use_time = False

    dir_path = os.path.dirname(os.path.realpath(__file__))
    
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
            finger_data = util.reshape_grayscale_data(finger_data)
            nbr_of_images = np.shape(finger_data)[0] # number of images to use from the original data set
            
            rotation_res = 1
            generator = data_generator(finger_data, finger_id, person_id, translation, rotation, nbr_of_images, rotation_res) # initialize data generator
            
            finger_id_gt_vt = np.load(dir_path + "/finger_id_gt_vt.npy")
            person_id_gt_vt = np.load(dir_path + "/person_id_gt_vt.npy")
            finger_data_gt_vt = np.load(dir_path + "/fingerprints_gt_vt.npy")
            translation_gt_vt = np.load(dir_path + "/translation_gt_vt.npy")
            rotation_gt_vt = np.load(dir_path + "/rotation_gt_vt.npy")
            finger_data_gt_vt = util.reshape_grayscale_data(finger_data_gt_vt)
            nbr_of_images_gt_vt = np.shape(finger_data_gt_vt)[0]
            
            generator.add_new_data(finger_data_gt_vt, finger_id_gt_vt, person_id_gt_vt, translation_gt_vt, rotation_gt_vt, nbr_of_images_gt_vt)
            pickle.dump(generator, output, pickle.HIGHEST_PROTOCOL)
    else:
        # Load generator
        with open('generator_data.pk1', 'rb') as input:
            generator = pickle.load(input)
             
    # parameters for training
    batch_size_train = 1500
    train_itr = 500000000000000000

    learning_rate = 0.00001
    momentum = 0.99
   
    # parameters for validation
    batch_size_val = 1500
    val_itr = 1000 # frequency in which to use validation data for computations
    
    # parameters for evaluation
    batch_size_test = 1500
    threshold = 0.5    
    thresh_step = 0.01
        
    dims = np.shape(generator.train_data[0])
    batch_sizes = [batch_size_train,batch_size_val,batch_size_test]
    image_dims = [dims[1],dims[2],dims[3]]
    
    save_itr = 100000 # frequency in which the model is saved
    
    tf.reset_default_graph()
    
    # if models exists use the latest existing one otherwise create a new one
    if not os.path.exists(output_dir + "checkpoint"):
        print("No previous model exists, creating a new one.")
        is_model_new = True
        meta_file_exists = False
        current_itr = 0 # current training iteration
        
        with tf.device(gpu_device_name):
             # create placeholders
            left_train,right_train,label_train,left_val,right_val,label_val,left_test,right_test = su.placeholder_inputs(image_dims,batch_sizes)
            handle = tf.placeholder(tf.string, shape=[],name="handle")
                
            left_train_output = sm.inference(left_train)            
            right_train_output = sm.inference(right_train)
            left_val_output = sm.inference(left_val, dropout = False)
            right_val_output = sm.inference(right_val, dropout = False)
            left_test_output = sm.inference(left_test, dropout = False)
            right_test_output = sm.inference(right_test, dropout = False)
            
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            margin = tf.constant(2.0) # margin for contrastive loss
            train_loss = su.contrastive_loss(left_train_output,right_train_output,label_train,margin)
            # add regularization terms to contrastive loss function
            for i in range(len(reg_losses)):
                train_loss += reg_losses[i]
            
            val_loss = su.contrastive_loss(left_val_output,right_val_output,label_val,margin)
            
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
        meta_file_exists = True
         
        with open(output_dir + "checkpoint","r") as file:
            line  = file.readline()
            words = re.split("/",line)
            model_file_name = words[-1][:-2]
            current_itr = int(re.split("-",model_file_name)[-1]) # current training iteration
            for file in os.listdir(output_dir):
                if file.endswith(".meta"):
                    meta_file_name = os.path.join(output_dir,file)
            saver = tf.train.import_meta_graph(meta_file_name)
        
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
                train_op = su.momentum_training(train_loss, learning_rate, momentum)
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
            
            # Summary setup
            conv1_filters = graph.get_tensor_by_name("conv1/kernel:0")
            nbr_of_filters_conv1 = sess.run(tf.shape(conv1_filters)[-1])
    
            conv2_filters = graph.get_tensor_by_name("conv2/kernel:0")
            hist_conv1 = tf.summary.histogram("hist_conv1", conv1_filters)
            hist_conv2 = tf.summary.histogram("hist_conv2", conv2_filters)
            conv1_filters = tf.transpose(conv1_filters, perm = [3,0,1,2])
            filter1 = tf.summary.image('Filter_1', conv1_filters, max_outputs=nbr_of_filters_conv1)
            conv1_bias = graph.get_tensor_by_name("conv1/bias:0")
            hist_bias1 = tf.summary.histogram("hist_bias1", conv1_bias)
            conv2_bias = graph.get_tensor_by_name("conv2/bias:0")
            hist_bias2 = tf.summary.histogram("hist_bias2", conv2_bias)
                
            summary_train_loss = tf.summary.scalar('training_loss', train_loss)
#            x_image = tf.summary.image('left_input', left_train)
            summary_op = tf.summary.merge([summary_train_loss, filter1, hist_conv1, hist_bias1, hist_conv2, hist_bias2])
            train_writer = tf.summary.FileWriter(output_dir + "/train_summary", graph=tf.get_default_graph())
             
        precision_over_time = []
        val_loss_over_time = []
        
        with tf.device(gpu_device_name):
            # Setup tensorflow's batch generator
            train_match_dataset = tf.data.Dataset.from_tensor_slices(generator.match_train)
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
        
        print("Starting training")
        start_time_train = time.time()
        # Training loop
        for i in range(1,train_itr + 1):
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
            if i % val_itr == 0:
                current_val_loss = 0
                b_sim_val_matching = np.repeat(np.ones((batch_size_val*int(val_match_dataset_length/batch_size_val),1)),generator.rotation_res,axis=0)
                b_sim_val_non_matching = np.repeat(np.zeros((batch_size_val*int(val_match_dataset_length/batch_size_val),1)),generator.rotation_res,axis=0)
                b_sim_full = np.append(b_sim_val_matching,b_sim_val_non_matching,axis=0)
                for j in range(int(val_match_dataset_length/batch_size_val)):
                    val_batch_matching = sess.run(next_element,feed_dict={handle:val_match_handle})
                    class_id_batch = generator.same_class(val_batch_matching)
                    for k in range(generator.rotation_res):
                        b_l_val,b_r_val = generator.get_pairs(generator.val_data[k],val_batch_matching) 
                        left_o,right_o,val_loss_value = sess.run([left_val_output,right_val_output, val_loss],feed_dict = {left_val:b_l_val, right_val:b_r_val, label_val:np.ones((batch_size_val,1))})
                        current_val_loss += val_loss_value
                        if j == 0 and k == 0:
                            left_full = left_o
                            right_full = right_o
                            class_id = class_id_batch
                        else:
                            left_full = np.vstack((left_full,left_o))
                            right_full = np.vstack((right_full,right_o))
                            class_id = np.vstack((class_id, class_id_batch))
                    
                for j in range(int(val_match_dataset_length/batch_size_val)):
                    val_batch_non_matching = sess.run(next_element,feed_dict={handle:val_non_match_handle})
                    for k in range(generator.rotation_res):
                        b_l_val,b_r_val = generator.get_pairs(generator.val_data[0],val_batch_non_matching) 
                        left_o,right_o,val_loss_value  = sess.run([left_val_output,right_val_output,val_loss],feed_dict = {left_val:b_l_val, right_val:b_r_val, label_val:np.zeros((batch_size_val,1))})
                        left_full = np.vstack((left_full,left_o))
                        right_full = np.vstack((right_full,right_o)) 
                        class_id_batch = generator.same_class(val_batch_non_matching)
                        class_id = np.vstack((class_id,class_id_batch))
                        current_val_loss += val_loss_value
                        
                val_loss_over_time.append(current_val_loss*batch_size_val/np.shape(b_sim_full)[0])
                precision, false_pos, false_neg, recall, fnr, fpr, inter_class_errors = sme.get_test_diagnostics(left_full,right_full, b_sim_full,threshold, class_id)
            
                if false_pos > false_neg:   # Can use inter_class_errors to tune the threshold further
                    threshold -= thresh_step
                else:
                    threshold += thresh_step
                precision_over_time.append(precision)

            train_writer.add_summary(summary, i)
            
            if use_time:
                elapsed_time = (time.time() - start_time_train)/60.0 # elapsed time in minutes since start of training 
                if elapsed_time >= int(argv[2]):
                    if meta_file_exists:
                        save_path = tf.train.Saver().save(sess,output_dir + "model",global_step=i+current_itr,write_meta_graph=False)
                    else:
                        save_path = tf.train.Saver().save(sess,output_dir + "model",global_step=i+current_itr)
                    print("Trained model after {} iterations and {} minutes saved in path: {}".format(i,elapsed_time,save_path))
                    break
                
            if i % save_itr == 0 or i == train_itr:
                if meta_file_exists:
                    save_path = tf.train.Saver().save(sess,output_dir + "model",global_step=i+current_itr,write_meta_graph=False)
                else:
                    save_path = tf.train.Saver().save(sess,output_dir + "model",global_step=i+current_itr)
                    meta_file_exists = True
                print("Trained model after {} iterations saved in path: {}".format(i,save_path))
        
        # Plot precision over time
        time_points = list(range(len(precision_over_time)))
        plt.plot(time_points, precision_over_time)
        plt.title("Precision over time")
        plt.xlabel("iteration")
        plt.ylabel("precision")
        plt.show()

        print("Suggested threshold from hyperparameter tuning: %f" % threshold)
        if len(precision_over_time) > 0:
            print("Final (last computed) precision: %f" % precision_over_time[-1])
        
        # Plot validation loss over time
        plt.figure()
        plt.plot(list(range(val_itr,val_itr*len(val_loss_over_time)+1,val_itr)),val_loss_over_time)
        plt.title("Validation loss (contrastive loss) over time")
        plt.xlabel("iteration")
        plt.ylabel("validation loss")
        plt.show()
        
if __name__ == "__main__":
     main(sys.argv[1:])