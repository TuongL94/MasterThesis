#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:46:32 2018

@author: exjobb
"""

import numpy as np
from data_generator_matrix import data_generator_matrix
import pickle
import os
import utilities as util



#data_path = os.path.dirname(os.path.realpath(__file__)) # directory of file being executed
data_path = "/home/PRECISE/exjobb/Documents/MasterThesis/data_manager/"


# Load fingerprint labels and data from file with names
#finger_id = np.load(data_path + "finger_id_mt_vt_112.npy")
#person_id = np.load(data_path + "person_id_mt_vt_112.npy")
#finger_data = np.load(data_path + "fingerprints_mt_vt_112.npy")
#
#class_breakpoints = np.load(data_path + "class_breakpoint.npy")
#matching_matrix = np.load(data_path + "matrix_gabor.npy")

finger_id = np.load(data_path + "finger_id_gabor.npy")
person_id = np.load(data_path + "person_id_gabor.npy")
finger_data = np.load(data_path + "fingerprints_gabor.npy")

class_breakpoints = np.load(data_path + "class_breakpoint.npy")
matching_matrix = np.load(data_path + "matrix_gabor.npy")


finger_data = util.reshape_grayscale_data(finger_data)
nbr_of_images = np.shape(finger_data)[0] # number of images to use from the original data set

matching_matrix = list(matching_matrix)

rotation_res = 1
generator = data_generator_matrix(finger_data, finger_id, person_id, matching_matrix, class_breakpoints, nbr_of_images, rotation_res) # initialize data generator

with open(data_path + "generator_data_gabor.pk1", "wb") as output:
    pickle.dump(generator, output, pickle.HIGHEST_PROTOCOL)