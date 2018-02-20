# -*- coding: utf-8 -*-
"""
Created on Thur Feb 15 18:25:36 2018

@author: Simon Nilsson
"""

import numpy as np
import tensorflow as tf
import os 
import utilities as util
import single_nn_train_mnist as sm


#def evaluate_mnist_siamese_network(eval_o,eval_labels):
#    nbr_correct = 0
#    estimated_digits = eval_o[0].argmax(axis=1)
#    for i in range(len(estimated_digits)):
#        if estimated_digits[i] == eval_labels[i]:
#            nbr_correct += 1
#    precision = nbr_correct/len(eval_labels)
#    return precision

def evaluate_mnist_siamese_network(eval_o,eval_labels):
    nbr_correct = 0
    estimated_digits = eval_o.argmax(axis=1)
    for i in range(len(estimated_digits)):
        if estimated_digits[i] == eval_labels[i]:
            nbr_correct += 1
    precision = nbr_correct/len(eval_labels)
    return precision

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
    
    batch_size = 10000
    
    output_dir = "/tmp/single_mnist_model/" # directory where the model is saved
    
    tf.reset_default_graph()
    
    if not os.path.exists(output_dir + ".meta"):
        print("No siamese model exists in " + output_dir)
        return
        
    else:
        print("Using existing model in the directory " + output_dir + " for evaluation")        
        saver = tf.train.import_meta_graph(output_dir + ".meta")        
        g = tf.get_default_graph()
        eval_data_place = g.get_tensor_by_name("data_eval:0")
        
        test_output = tf.get_collection("test_output")[0]
        softmax_layer = tf.nn.softmax(test_output,axis=0)
        
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(output_dir))
            counter = 0
            b_data,_ = sm.batch_mnist(batch_size, counter, eval_labels, eval_data)
            eval_o,prob = sess.run([test_output,softmax_layer],feed_dict = {eval_data_place:b_data})

            precision = evaluate_mnist_siamese_network(prob, eval_labels)
            print("Precision: %f " % precision)
      
if __name__ == "__main__":
    tf.app.run()
    
    
    
    
    

        