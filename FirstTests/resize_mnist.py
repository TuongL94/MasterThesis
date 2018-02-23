#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:00:22 2018

@author: simon
"""

import numpy as np
import tensorflow as tf
import utilities as util
from skimage.transform import resize
import os

# Load original images
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = util.reshape_grayscale_data(mnist.train.images) # Returns np.array
test_data = util.reshape_grayscale_data(mnist.test.images)
val_data = util.reshape_grayscale_data(mnist.validation.images)

# Resize mnist to fingerprint size (192,192)
resized_train_images = np.zeros((train_data.shape[0], 192,192,1),dtype='float32')
for i in range(train_data.shape[0]):
    resized_train_images[i] = resize(train_data[i], (192, 192))#, anti_aliasing=False)
    
resized_test_images = np.zeros((test_data.shape[0], 192,192,1),dtype='float32')
for i in range(test_data.shape[0]):
    resized_test_images[i] = resize(test_data[i], (192,192))
    
resized_val_images = np.zeros((val_data.shape[0], 192,192,1),dtype='float32')
for i in range(val_data.shape[0]):
    resized_val_images[i] = resize(val_data[i], (192, 192))


# Save resized images as numpy arrays
dir_path = os.path.dirname(os.path.realpath(__file__)) # directory of file being executed
filename_1 = dir_path + "/resized_train_mnist" 
filename_2 = dir_path + "/resized_test_mnist" 
filename_3 = dir_path + "/resized_val_mnist"
np.save(filename_1,resized_train_images)
np.save(filename_2,resized_test_images)
np.save(filename_3,resized_val_images)



