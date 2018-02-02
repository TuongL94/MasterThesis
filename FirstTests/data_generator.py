# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:24:03 2018

@author: Tuong Lam
"""

import numpy as np

class data_generator:
    
    def __init__(self,images,labels):
        nbr_of_images,dim_squarred = np.shape(images)
        self.trainsize = 10000
        dim = int(np.sqrt(dim_squarred))
        self.labels = labels[0:self.trainsize]
        tmp_images = images.reshape((nbr_of_images,dim,dim,1))
        self.images = tmp_images[0:self.trainsize,:,:,:]
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
        
    def prep_eval_data(self,eval_data,eval_labels):
        nbr_of_images,dim_squarred = np.shape(eval_data)
        dim = int(np.sqrt(dim_squarred))
        nbr_of_image_pairs = int(nbr_of_images/2)
        eval_data_moded = eval_data.reshape((nbr_of_images,dim,dim,1))
        left = []
        right = []
        sim = np.zeros(nbr_of_image_pairs)
        for i in range(nbr_of_image_pairs):
            left.append(eval_data_moded[i,:,:,:])
            right.append(eval_data_moded[nbr_of_image_pairs+i,:,:,:])
            if(eval_labels[i] == eval_labels[nbr_of_image_pairs + i]):
                sim[i] = 1
                
        return np.array(left),np.array(right),sim