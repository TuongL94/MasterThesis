# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:12:05 2018

@author: Tuong Lam
"""

import numpy as np
import scipy.linalg as sl
import random
import tensorflow as tf
import os
import pickle

def print_all_global_variables():
    global_vars = tf.global_variables()
    for i in range(len(global_vars)):
        print(global_vars[i])
    
def get_nbr_of_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    return total_parameters

def l2_normalize(input_array):
    """ L2-normalizes a 1D or 2D array along first dimension
    
    Input:
    input_array - the array to l2-normalize
    Returns: input_array l2-normalized along first dimension
    """
    dims = np.shape(input_array)
    if len(dims) == 1:
        return input_array/sl.norm(input_array)
    else:
        max_length = -1
        for i in range(dims[0]):
            current_length = sl.norm(input_array[i,:])            
            if current_length > max_length:
                max_length = current_length                   
        return input_array/max_length
                    
def reshape_grayscale_data(input_data, *dims):
    """ Reshapes 2D grayscale data to 4D.
    
    Input:
    input_data - 2D grayscale data
    dims - tuple containing the height and width of the original images in the non-square case.
    If the original images are square this parameter can be omitted.
    Returns:
    input_data_moded - 4D grayscale data
    """
    if len(dims) == 1:
        nbr_of_images, _ = np.shape(input_data)
        input_data_moded = input_data.reshape((nbr_of_images,dims[0][0],dims[0][1],1))
    else:
        nbr_of_images,dim_squarred = np.shape(input_data)
        dim = int(np.sqrt(dim_squarred))
        input_data_moded = input_data.reshape((nbr_of_images,dim,dim,1))
    return input_data_moded
        
def image_standardization(images):
    dims = np.shape(images)
    means = np.mean(images, axis=(1,2,3), keepdims=True)
    stds = np.std(images, axis=(1,2,3), keepdims=True)
    adjusted_stds = np.maximum(stds,1.0/np.sqrt(dims[1]*dims[2])) # avoid division by zero
    images = (images - means)/stds 
    return images

def rand_assign_pair(left,right,image_1,image_2):
    """ Appends images of an image pair randomly to two lists.
    
    Input:
    left - list to which one image of an image pair will be appended
    right - list to which one image of an image pair will be appended
    image_1 - first image pair
    image_2 - second image pair
    """
    if random.uniform(0,1) < 0.5:
        left.append(image_1)
        right.append(image_2)
    else:
        left.append(image_2)
        right.append(image_1)
    
def get_no_matching_subset(data_generator, data_path, new_generator_name):
    if not os.path.exists(data_path + new_generator_name):
        with open(data_path + new_generator_name, "wb") as output:
            size_match_train = len(data_generator.match_train)
            size_match_val = len(data_generator.match_val)
            size_match_test = len(data_generator.match_test)
            
            # shuffle non matching pairs
            np.random.shuffle(data_generator.no_match_train)
            np.random.shuffle(data_generator.no_match_val)
            np.random.shuffle(data_generator.no_match_test)
            
            # get subset of non matching pairs
            data_generator.no_match_train = data_generator.no_match_train[0:10*size_match_train]
            data_generator.no_match_val = data_generator.no_match_val[0:10*size_match_val]
            data_generator.no_match_test = data_generator.no_match_test[0:10*size_match_test]
            
            pickle.dump(data_generator, output, pickle.HIGHEST_PROTOCOL)
            