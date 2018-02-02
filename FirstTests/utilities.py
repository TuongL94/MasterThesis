# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:12:05 2018

@author: Tuong Lam
"""

import numpy as np
import scipy.linalg as sl

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
    
def reshape_grayscale_data(input_data):
    """ Reshapes 2D grayscale data to 4D.
    
    Input:
    input_data - 2D grayscale data
    Returns:
    input_data_moded - 4D grayscale data
    """
    nbr_of_images,dim_squarred = np.shape(input_data)
    dim = int(np.sqrt(dim_squarred))
    input_data_moded = input_data.reshape((nbr_of_images,dim,dim,1))
    return input_data_moded

def prep_eval_data(eval_data,eval_labels):
    eval_data_moded = reshape_grayscale_data(eval_data)
    dims = np.shape(eval_data_moded)
    nbr_of_image_pairs = int(dims[0]/2)
    
    left = []
    right = []
    sim = np.zeros(nbr_of_image_pairs)
    for i in range(nbr_of_image_pairs):
        left.append(eval_data_moded[i,:,:,:])
        right.append(eval_data_moded[nbr_of_image_pairs+i,:,:,:])
        if(eval_labels[i] == eval_labels[nbr_of_image_pairs + i]):
            sim[i] = 1
            
    return np.array(left),np.array(right),sim