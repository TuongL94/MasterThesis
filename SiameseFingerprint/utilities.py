# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:12:05 2018

@author: Tuong Lam
"""

import numpy as np
import scipy.linalg as sl
import random

def l2_normalize(input_array):
    """ L2-normalizes a 1D or 2D array along first dimension
    
    Input:
    input_array - the array to l2-normalize
    Returns: input_array l2-normalized along first dimension
    """
    dims = np.shape(input_array)
    if len(dims) == 1:
        # Rescaled the data to (0,1) interval
#        max_in = input_array.max(1)
#        min_in = input_array.min(1)
#        return (input_array - min_in)/(max_in - min_in)
        # Scale max to 1
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
    """ Shuffles all elements in the specified list, the permutation is the same for all elements of the list.
    
    Input:
    data_list - list containing all elements to be shuffled
    Returns:
    shuffled_data_list - data_list permuted, all elements are permuted in the same manner
    """
    index_shuf = list(range(len(data_list[-1])))
    random.shuffle(index_shuf)
    shuffled_data_list = []
    for i in range(len(data_list)):
        shuffled_data_list.append([])
    for i in index_shuf:
        for j in range(len(shuffled_data_list)):
            shuffled_data_list[j].append(data_list[j][i])
    
    return shuffled_data_list

#def generate_pair(images,shift_ind,sim_ind):
#    new_finger = random.randint(0,len(shift_ind)-2)
#    index_finger = random.randint(shift_ind[new_finger], shift_ind[new_finger+1])
#    
#    if sim_ind == 0:    # Make unmatched pair
#        while True:
#            index_non_match = random.randint(shift_ind[0], shift_ind[-1])
#            if not shift_ind[new_finger] <= index_non_match <= shift_ind[new_finger+1]:
#                break
#        return images[index_finger],images[index_non_match], [0]
#    else:           # Make matching pair
#        index_match = random.randint(shift_ind[new_finger], shift_ind[new_finger+1])
#        return images[index_finger],images[index_match], [1]
    
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
    