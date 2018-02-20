# -*- coding: utf-8 -*-
"""
Created on Thur Feb 15 11:22:22 2018

@author: Simon Nilsson
"""


import single_nn_model_mnist as sm
import numpy as np
import tensorflow as tf
import os 
import utilities as util


def batch_mnist(batch_size, counter, train_labels, train_data):
    b_data = train_data[counter:counter+batch_size,:,:,:]
    b_labels = train_labels[counter:counter+batch_size]
    return b_data,b_labels
    

def main(unused_argv):
    
    # Load mnist training and eval data and perform necessary data reshape
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    mnist_train_data = util.reshape_grayscale_data(mnist.train.images) # Returns np.array
    mnist_train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    
    output_dir = "/tmp/single_mnist_model/" # directory where the model will be saved
    
    nbr_of_training_images = 55000 # number of images to use from the training data set
        
    # parameters for training
    batch_size = 100
    train_iter = 20000
    learning_rate = 0.001
    momentum = 0.99
    
    image_dims = np.shape(mnist_train_data)
    placeholder_dims = [batch_size, image_dims[1], image_dims[2], image_dims[3]] 
    
    # parameters for evaluation
    batch_size_test = 10000
    
    tf.reset_default_graph()
    
    # if models exists use the existing one otherwise create a new one
    if not os.path.exists(output_dir + ".meta"):
        print("No previous model exists, creating a new one.")
        is_model_new = True

         # create placeholders
        data,label,test_data = sm.placeholder_inputs(placeholder_dims,batch_size_test)
            
        train_output = sm.inference(data)            
        test_output = sm.inference(test_data)
        onehot_labels = tf.one_hot(indices = tf.cast(label, tf.int32), depth = 10)
        onehot_labels = tf.squeeze(onehot_labels)
        loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = train_output)
        
        tf.add_to_collection("loss",loss)
        tf.add_to_collection("train_output",train_output)
        tf.add_to_collection("test_output",test_output)
        saver = tf.train.Saver()
        
    else:
        print("Using existing model in the directory " + output_dir)
        is_model_new = False
        
        saver = tf.train.import_meta_graph(output_dir + ".meta")
        g = tf.get_default_graph()
        data = g.get_tensor_by_name("data:0")
        label = g.get_tensor_by_name("label:0")
        loss = tf.get_collection("loss")[0]
        train_output = tf.get_collection("train_output")[0]
        
    with tf.Session() as sess:
        if is_model_new:
            train_op = sm.training(loss, learning_rate, momentum)
            sess.run(tf.global_variables_initializer()) # initialize all trainable parameters
            tf.add_to_collection("train_op",train_op)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(output_dir))
            train_op = tf.get_collection("train_op")[0]
            
#        graph = tf.get_default_graph()
#        conv1_layer = graph.get_tensor_by_name("conv_layer_1/kernel:0")
#        nbr_of_filters_conv1 = sess.run(tf.shape(conv1_layer)[-1])
#
#        conv2_layer = graph.get_tensor_by_name("conv_layer_2/kernel:0")
#        hist_conv1 = tf.summary.histogram("hist_conv1", conv1_layer)
#        hist_conv2 = tf.summary.histogram("hist_conv2", conv2_layer)
#        conv1_layer = tf.transpose(conv1_layer, perm = [3,0,1,2])
#        filter1 = tf.summary.image('Filter_1', conv1_layer, max_outputs=nbr_of_filters_conv1)
#        conv1_layer = tf.transpose(conv1_layer, perm = [1,2,3,0])
##        conv2_layer = tf.transpose(conv2_layer, perm = [3,0,1,2])
##        filter2 = tf.summary.image('Filter_2', conv2_layer, max_outputs=32)
#        bias_conv1 = graph.get_tensor_by_name("conv_layer_1/bias:0")
#        hist_bias1 = tf.summary.histogram("hist_bias1", bias_conv1)
#        bias_conv2 = graph.get_tensor_by_name("conv_layer_2/bias:0")
#        hist_bias2 = tf.summary.histogram("hist_bias2", bias_conv2)
#            
#        summary_op = tf.summary.scalar('training_loss', loss)
#        x_image = tf.summary.image('input', data)
#        summary_op = tf.summary.merge([summary_op, x_image, filter1, hist_conv1, hist_conv2, hist_bias1, hist_bias2])
#        # Summary setup
#        writer = tf.summary.FileWriter(output_dir + "/summary", graph=tf.get_default_graph())
        
        counter = 0
        for i in range(1,train_iter + 1):
            b_data, b_labels = batch_mnist(batch_size, counter, mnist_train_labels, mnist_train_data)
            _,loss_value,train_o = sess.run([train_op, loss, train_output],feed_dict={data:b_data, label:b_labels})
            if i % 100 == 0:
                print("Iteration %d: loss = %.5f" % (i, loss_value))
                
#            writer.add_summary(summary, i)
            counter = (counter + batch_size) % nbr_of_training_images

        save_path = tf.train.Saver().save(sess,output_dir)
        print("Trained model saved in path: %s" % save_path)
    
    
if __name__ == "__main__":
    tf.app.run()
    

    