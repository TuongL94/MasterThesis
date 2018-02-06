# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:24:03 2018

@author: Tuong Lam
"""

import numpy as np
import random
import utilities as util


class data_generator:

    def __init__(self, images, labels,train_size):
        self.nbr_of_classes = len(np.unique(labels))
        self.labels = labels[0:train_size]
        self.images = images[0:train_size,:,:,:]
        
        # Sort images so that they come in 0..9
        # This is maybe better to do before pasing to this function
        self.images = [x for _, x in sorted(zip(self.labels, self.images), key=lambda pair: pair[0])]
        self.labels.sort()
        self.digit = []
        for i in range(self.nbr_of_classes):
            self.digit.append(np.where(self.labels == i)[0][0])
        self.digit.append(len(self.labels) - 1)

    def gen_pair_batch(self,batch_size):
        left = []
        right = []
        sim = []
        count = 0
        
        for i in range(self.nbr_of_classes):
            nbr_each_digit = int(batch_size/20);
            for j in range(nbr_each_digit):
                l = random.randint(self.digit[i], self.digit[i+1])
                r = random.randint(self.digit[i], self.digit[i+1])
                left.append(self.images[l])
                right.append(self.images[r])
                sim.append([1])
                count += 1
            for j in range(nbr_each_digit):
                # Generate random index of current digit i
                rnd_current_digit = random.randint(self.digit[i], self.digit[i+1])
                # Generate random index of digit not being i
                while True:
                    rnd_other_digit = random.randint(self.digit[0], self.digit[-1])
                    if not self.digit[i] <= rnd_other_digit <= self.digit[i+1]:
                        break
                      
                # Put the pair in random left/right
                if random.uniform(0,1) < 0.5:
                    left.append(self.images[rnd_current_digit])
                    right.append(self.images[rnd_other_digit])
                else:
                    left.append(self.images[rnd_other_digit])
                    right.append(self.images[rnd_current_digit])                
                    
                sim.append([0])
                count += 1
                
        # Generate remaining pairs in the batch
        # Make matching pairs every second time
        mat = 0
        while count < batch_size:
            new_digit = random.randint(0,self.nbr_of_classes-1)
            index_digit = random.randint(self.digit[new_digit], self.digit[new_digit+1])
            if mat == 0:    # Make unmatched pair
                while True:
                    index_non_match = random.randint(self.digit[0], self.digit[-1])
                    if not self.digit[new_digit] <= index_non_match <= self.digit[new_digit+1]:
                        break
                left.append(self.images[index_digit])
                right.append(self.images[index_non_match])
                sim.append([0])
                mat = 1 # Set next pair to be matching
            else:           # Make matching pair
                index_match = random.randint(self.digit[new_digit], self.digit[new_digit+1])
                left.append(self.images[index_digit])
                right.append(self.images[index_match])
                sim.append([1])
                mat = 0 #Set next pair to be non matching
            count += 1
                
        # Shuffle the data pairs        
        data_list = [left,right,sim]
        shuffled_data_list = util.shuffle_data(data_list)
            
        return np.array(shuffled_data_list[0]),np.array(shuffled_data_list[1]),shuffled_data_list[2]
