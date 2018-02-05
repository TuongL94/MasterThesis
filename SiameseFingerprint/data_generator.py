# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:24:03 2018

@author: Tuong Lam
"""

import numpy as np
import random
from random import shuffle


class data_generator:

    def __init__(self, images, labels,train_size):
        self.labels = labels[0:train_size]
        self.images = images[0:train_size,:,:,:]
        self.ind = 0
        self.trainsize = train_size
        
        # Sort images so that they come in 0..9
        # This is maybe better to do before pasing to this function
        self.images = [x for _, x in sorted(zip(self.labels, self.images), key=lambda pair: pair[0])]
        self.labels.sort()
        self.digit = []
        for i in range(10):
            self.digit.append(np.where(self.labels == i)[0][0])
        self.digit.append(len(self.labels) - 1)
    
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

    def gen_pair_batch(self,batch_size):
        left = []
        right = []
        sim = []
        count = 0
        
        for i in range(10):
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
            new_digit = random.randint(0,9)
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
        index_shuf = list(range(len(sim)))
        shuffle(index_shuf)
        temp_l = []
        temp_r = []
        temp_s = []
        for i in index_shuf:
            temp_l.append(left[i])
            temp_r.append(right[i])
            temp_s.append(sim[i])
        left = temp_l
        right = temp_r
        sim = temp_s
            
        return np.array(left),np.array(right),sim
