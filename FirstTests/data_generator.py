# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:24:03 2018

@author: Tuong Lam
"""

import numpy as np
import random
import utilities as util
import math

class data_generator:

    def __init__(self, train_data, train_labels, val_data,val_labels, eval_data,eval_labels,data_sizes):
        self.nbr_of_classes = len(np.unique(train_labels))
        self.train_data = train_data[0:data_sizes[0],:,:,:]
        self.train_labels = train_labels[0:data_sizes[0]]
        self.val_data = val_data[0:data_sizes[1],:,:,:]
        self.val_labels = val_labels[0:data_sizes[1]]
        self.eval_data = eval_data[0:data_sizes[2],:,:,:]
        self.eval_labels = eval_labels[0:data_sizes[2]]
        
        # Sort all data in ascending order of class labels (0-9)
        self.train_data = [x for _, x in sorted(zip(self.train_labels, self.train_data), key=lambda pair: pair[0])]
        self.train_labels.sort()
        self.val_data = [x for _, x in sorted(zip(self.val_labels, self.val_data), key=lambda pair: pair[0])]
        self.val_labels.sort()
        self.eval_data = [x for _, x in sorted(zip(self.eval_labels, self.eval_data), key=lambda pair: pair[0])]
        self.eval_labels.sort()
        
        self.train_breakpoints = []
        for i in range(self.nbr_of_classes):
            self.train_breakpoints.append(np.where(self.train_labels == i)[0][0])
        self.train_breakpoints.append(len(self.train_labels) - 1)
        
        self.val_breakpoints = []
        for i in range(self.nbr_of_classes):
            self.val_breakpoints.append(np.where(self.val_labels == i)[0][0])
        self.val_breakpoints.append(len(self.val_labels) - 1)
        
        self.eval_breakpoints = []
        for i in range(self.nbr_of_classes):
            self.eval_breakpoints.append(np.where(self.eval_labels == i)[0][0])
        self.eval_breakpoints.append(len(self.eval_labels) - 1)
        
        
        self.all_match_train, self.all_no_match_train = self.all_combinations(self.train_data,self.train_labels,self.train_breakpoints)
        self.all_match_val, self.all_no_match_val = self.all_combinations(self.val_data,self.val_labels,self.val_breakpoints)
        self.all_match_eval, self.all_no_match_eval = self.all_combinations(self.eval_data,self.eval_labels,self.eval_breakpoints)
        
    def all_combinations(self,data,labels,data_breakpoints):
        match = [] # matching pair indices
        no_match = [] # non-matching pair indices 
        
        for i in range(len(data_breakpoints)-1):
            for k in range(data_breakpoints[i+1]-data_breakpoints[i]):
                for j in range(data_breakpoints[i]+k+1, data_breakpoints[i+1]):

                    if labels[i]+k == labels[j]:
                        match.append([data_breakpoints[i]+k, j])
                    else:
                        no_match.append([data_breakpoints[i]+k, j])
                    
                for n in range(data_breakpoints[i+1], np.shape(data)[0]):
                    no_match.append([data_breakpoints[i]+k, n])
            
        no_match = np.array(no_match)
        np.random.shuffle(no_match)
        return match,no_match
    
#    def gen_batch(self, batch_size, training = 1):
#        """ Generates a batch with matched and non-matched pairs of images.
#        
#        Input:
#        batch_size - the size of the batch
#        training - parameter to determine if the generated batch should be used for training,
#        if training is set to 1 the batch is used for training otherwise it is used for validation.
#        Returns: three numpy arrays where the first two contain one image from randomly selected image pairs respectively.
#        The last array indicates if corresponding pairs in the first two arrays are matching or non-matching
#        pairs, if the pairs match the corresponding element in the last array is 1, otherwise 0.
#        """
#        left = []
#        right = []
#        sim = []
#        
#        if training == 1:
#            current_matching_set = self.match_train
#            current_non_matching_set = self.no_match_train
#        else:
#            current_matching_set = self.match_val
#            current_non_matching_set = self.no_match_val
#            
#        switch_match = 0
#        for i in range(batch_size):
#            if switch_match == 0:
#                rnd_match = current_matching_set[random.randint(0,len(current_matching_set)-1)]
#                util.rand_assign_pair(left,right,self.images[rnd_match[0]],self.images[rnd_match[1]])
#                sim.append([1])
#                switch_match = 1
#            else:
#                rnd_no_match = current_non_matching_set[random.randint(0,len(current_non_matching_set)-1)]
#                util.rand_assign_pair(left,right,self.images[rnd_no_match[0]],self.images[rnd_no_match[1]])
#                sim.append([0])
#                switch_match = 0
#        
#        left, right, sim =  util.shuffle_data([left, right, sim])    
#        
#        return np.array(left), np.array(right), np.array(sim)
#
#    def gen_eval_batch(self,batch_size):
#        left = []
#        right = []
#        sim = []
#        count = 0
#        
#        for i in range(self.nbr_of_classes):
#            nbr_each_digit = int(batch_size/20);
#            for j in range(nbr_each_digit):
#                l = random.randint(self.digit[i], self.digit[i+1])
#                r = random.randint(self.digit[i], self.digit[i+1])
#                left.append(self.images[l])
#                right.append(self.images[r])
#                sim.append([1])
#                count += 1
#            for j in range(nbr_each_digit):
#                # Generate random index of current digit i
#                rnd_current_digit = random.randint(self.digit[i], self.digit[i+1])
#                # Generate random index of digit not being i
#                while True:
#                    rnd_other_digit = random.randint(self.digit[0], self.digit[-1])
#                    if not self.digit[i] <= rnd_other_digit <= self.digit[i+1]:
#                        break
#                      
#                # Put the pair in random left/right
#                if random.uniform(0,1) < 0.5:
#                    left.append(self.images[rnd_current_digit])
#                    right.append(self.images[rnd_other_digit])
#                else:
#                    left.append(self.images[rnd_other_digit])
#                    right.append(self.images[rnd_current_digit])                
#                    
#                sim.append([0])
#                count += 1
#                
#        # Generate remaining pairs in the batch
#        # Make matching pairs every second time
#        mat = 0
#        while count < batch_size:
#            new_digit = random.randint(0,self.nbr_of_classes-1)
#            index_digit = random.randint(self.digit[new_digit], self.digit[new_digit+1])
#            if mat == 0:    # Make unmatched pair
#                while True:
#                    index_non_match = random.randint(self.digit[0], self.digit[-1])
#                    if not self.digit[new_digit] <= index_non_match <= self.digit[new_digit+1]:
#                        break
#                left.append(self.images[index_digit])
#                right.append(self.images[index_non_match])
#                sim.append([0])
#                mat = 1 # Set next pair to be matching
#            else:           # Make matching pair
#                index_match = random.randint(self.digit[new_digit], self.digit[new_digit+1])
#                left.append(self.images[index_digit])
#                right.append(self.images[index_match])
#                sim.append([1])
#                mat = 0 #Set next pair to be non matching
#            count += 1
#                
#        # Shuffle the data pairs        
#        data_list = [left,right,sim]
#        shuffled_data_list = util.shuffle_data(data_list)
#            
#        return np.array(shuffled_data_list[0]),np.array(shuffled_data_list[1]),np.array(shuffled_data_list[2])
    
#    def two_split_array(self,input_array,percentage):
#        length = len(input_array)
#        split_ind = math.floor(length*percentage)
#        
#        first_split = input_array[0:split_ind+1]
#        second_split = input_array[split_ind+1:]
#        return first_split,second_split
