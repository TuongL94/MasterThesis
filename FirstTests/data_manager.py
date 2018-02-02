# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:53:44 2018

@author: Tuong Lam
"""

import numpy as np
import utilities as util

class data_manager:
    
    def __init__(self,train_images,train_labels,eval_images,eval_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.eval_images = eval_images
        self.eval_labels = eval_labels
        
        self.ind = 0
        
    def gen_batch(self,batch_size):
        count = 0
        left = []
        right = []
        sim = []
        
        while self.ind < self.trainsize - 2 and count < batch_size:
            left.append(self.images[self.ind])
            right.append(self.images[self.ind+1])
            if self.labels[self.ind] == self.labels[self.ind+1]:
                sim.append([1])
            else:
                sim.append([0])
            self.ind += 1
            count += 1
            
        for i in range(batch_size - count):
            left.append(self.images[i])
            right.append(self.images[i+1])
            if self.labels[i] == self.labels[i+1]:
                sim.append([1])
            else:
                sim.append([0])
            self.ind += 1
            
        return np.array(left),np.array(right),sim
        
    def prep_eval_data(eval_data,eval_labels):
        eval_data_moded = util.reshape_grayscale_data(eval_data)
        dims = np.shape(eval_data_moded)
        nbr_of_image_pairs = dims[0]/2
        
        left = []
        right = []
        sim = np.zeros(nbr_of_image_pairs)
        for i in range(nbr_of_image_pairs):
            left.append(eval_data_moded[i,:,:,:])
            right.append(eval_data_moded[nbr_of_image_pairs+i,:,:,:])
            if(eval_labels[i] == eval_labels[nbr_of_image_pairs + i]):
                sim[i] = 1
                
        return np.array(left),np.array(right),sim
