# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:24:03 2018

@author: Tuong Lam
"""

import numpy as np
import utilities as util

class data_generator:

    def __init__(self,train_data,train_labels,val_data,val_labels,test_data,test_labels,data_sizes):
        self.nbr_of_classes = len(np.unique(train_labels))
        self.train_data = train_data[0:data_sizes[0],:,:,:]
        self.train_labels = train_labels[0:data_sizes[0]]
        self.val_data = val_data[0:data_sizes[1],:,:,:]
        self.val_labels = val_labels[0:data_sizes[1]]
        self.test_data = test_data[0:data_sizes[2],:,:,:]
        self.test_labels = test_labels[0:data_sizes[2]]
        
        # Sort all data in ascending order of class labels (0-9)
        self.train_data = [x for _, x in sorted(zip(self.train_labels, self.train_data), key=lambda pair: pair[0])]
        self.train_labels.sort()
        self.val_data = [x for _, x in sorted(zip(self.val_labels, self.val_data), key=lambda pair: pair[0])]
        self.val_labels.sort()
        self.test_data = [x for _, x in sorted(zip(self.test_labels, self.test_data), key=lambda pair: pair[0])]
        self.test_labels.sort()
        
        self.train_breakpoints = []
        for i in range(self.nbr_of_classes):
            self.train_breakpoints.append(np.where(self.train_labels == i)[0][0])
        self.train_breakpoints.append(len(self.train_labels))
        
        self.val_breakpoints = []
        for i in range(self.nbr_of_classes):
            self.val_breakpoints.append(np.where(self.val_labels == i)[0][0])
        self.val_breakpoints.append(len(self.val_labels))
        
        self.test_breakpoints = []
        for i in range(self.nbr_of_classes):
            self.test_breakpoints.append(np.where(self.test_labels == i)[0][0])
        self.test_breakpoints.append(len(self.test_labels))
        
        self.all_match_train, self.all_non_match_train = self.all_combinations(self.train_data,self.train_breakpoints)
        self.all_match_val, self.all_non_match_val = self.all_combinations(self.val_data,self.val_breakpoints)
        self.all_match_test, self.all_non_match_test = self.all_combinations(self.test_data,self.test_breakpoints)
        
    def all_combinations(self,data,data_breakpoints):
        match = [] # matching pair indices
        no_match = [] # non-matching pair indices 
        
        for i in range(len(data_breakpoints)-1):
            for k in range(data_breakpoints[i+1]-data_breakpoints[i]):
                for j in range(data_breakpoints[i]+k+1, data_breakpoints[i+1]):
                        match.append([data_breakpoints[i]+k, j])
                    
                for n in range(data_breakpoints[i+1], np.shape(data)[0]):
                    no_match.append([data_breakpoints[i]+k, n])

        return np.array(match),np.array(no_match)
        
    def get_pairs(self,data,pair_list):
        left_pairs = np.take(data,pair_list[:,0],axis=0)
        right_pairs = np.take(data,pair_list[:,1],axis=0)
        return left_pairs,right_pairs

    def gen_batch(self,batch_size,training=1):
        """ Generates a batch with matched and non-matched pairs of images.
        
        Input:
        batch_size - the size of the batch
        training - parameter to determine if the generated batch should be used for training,
        if training is set to 1 the batch is used for training otherwise it is used for validation.
        Returns: three numpy arrays where the first two contain one image from randomly selected image pairs respectively.
        The last array indicates if corresponding pairs in the first two arrays are matching or non-matching
        pairs, if the pairs match the corresponding element in the last array is 1, otherwise 0.
        """
        left = []
        right = []
        sim = []
#        sim = np.zeros(batch_size)
        
        if training == 1:
            current_matching_set = self.all_match_train
            current_non_matching_set = self.all_non_match_train
        else:
            current_matching_set = self.all_match_val
            current_non_matching_set = self.all_non_match_val
            
        switch_match = 0
        for i in range(batch_size):
            if switch_match == 0:
                rnd_match = current_matching_set[np.random.randint(0,len(current_matching_set)-1)]
                util.rand_assign_pair(left,right,self.train_data[rnd_match[0]],self.train_data[rnd_match[1]])
                sim.append([1])
                switch_match = 1
            else:
                rnd_no_match = current_non_matching_set[np.random.randint(0,len(current_non_matching_set)-1)]
                util.rand_assign_pair(left,right,self.train_data[rnd_no_match[0]],self.train_data[rnd_no_match[1]])
                sim.append([0])
                switch_match = 0
        
        left, right, sim =  util.shuffle_data([left, right, sim])    
        
        return np.array(left), np.array(right), np.array(sim)