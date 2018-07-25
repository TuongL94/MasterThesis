# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:12:05 2018

@author: Tuong Lam & Simon Nilsson 
"""

import numpy as np
import scipy.linalg as sl
import random
import tensorflow as tf
#import os
#import pickle
import matplotlib.pyplot as plt

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
        
        
        
def save_evaluation_metrics(metrics, file_path):
    nbr_of_metrics = len(metrics)
    with open(file_path, "a") as file:
        for i in range(nbr_of_metrics):
            file.write(repr(metrics[i]) + " ")
        file.write("\n")
        
def get_evaluation_metrics_vals(file_path):
    fpr_vals = []
    fnr_vals = []
    recall_vals = []
    acc_vals = []
    
    with open(file_path, "rb") as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line  = lines[i]
            line = line.split()
            fpr_vals.append(float(line[0]))
            fnr_vals.append(float(line[1]))
            recall_vals.append(float(line[2]))
            acc_vals.append((float(line[3])))
    
    return fpr_vals, fnr_vals, recall_vals, acc_vals

def plot_evaluation_metrics(thresholds, fpr_vals, fnr_vals, recall_vals, acc_vals):
    plt.figure()
    plt.plot(thresholds, fpr_vals, "b", label="FPR")
    plt.plot(thresholds, fnr_vals, "r", label="FNR")
    plt.plot(thresholds, recall_vals, label="recall")
    plt.xlabel("threshold")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    
    plt.figure()
    plt.plot(thresholds, acc_vals, label="accuracy")
    plt.xlabel("threshold")
    plt.ylabel("accuracy")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    
    plt.figure()
    plt.semilogx(fpr_vals,recall_vals)
    plt.xlabel("FPR")
    plt.ylabel("Recall")
    plt.show()
    
def get_separation_distance_hist(l2_distances_sim, l2_distances_no_sim):
    plt.hist(l2_distances_sim, bins="auto", alpha=0.5, label="similar", color="g")
    plt.hist(l2_distances_no_sim, bins="auto", alpha=0.5, label="dissimilar", color="r")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    