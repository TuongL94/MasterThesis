# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:12:05 2018

@author: Tuong Lam
"""

import numpy as np
import scipy.linalg as sl
from random import shuffle

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

def shuffle_data(data_list):
    index_shuf = list(range(len(data_list[-1])))
    shuffle(index_shuf)
    shuffled_data_list = []
    for i in range(len(data_list)):
        shuffled_data_list.append([])
    for i in index_shuf:
        for j in range(len(shuffled_data_list)):
            shuffled_data_list[j].append(data_list[j][i])
    
    return shuffled_data_list