# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:24:03 2018

@author: Tuong Lam
"""

import numpy as np
import random
import utilities as util

class data_generator:

    def __init__(self, images, finger_id, person_id, train_size):
        self.finger_id = finger_id[0:train_size]
        self.person_id = person_id[0:train_size]
        self.images = images[0:train_size,:,:,:]
        self.shift_idx = []
        idx_counter = 0
        nbr_of_persons = person_id[-1]
        for i in range(nbr_of_persons):
            finger_counter = 0
            while idx_counter < len(person_id):
                if not i+1 == self.person_id[idx_counter]:
                    break
                if not finger_counter == self.finger_id[idx_counter]:
                    finger_counter = self.finger_id[idx_counter]
                    self.shift_idx.append(idx_counter)
                
                idx_counter += 1
                
        self.shift_idx.append(len(self.person_id) - 1)

    def gen_pair_batch(self,batch_size):
        left = []
        right = []
        sim = []
        count = 0
        
        nbr_fingerprints = len(self.finger_id) - 1
        for i in range(nbr_fingerprints):
            nbr_each_fingerprint = int(batch_size/nbr_fingerprints/2);
            for j in range(nbr_each_fingerprint):
                l = random.randint(self.shift_idx[i], self.shift_idx[i+1])
                r = random.randint(self.shift_idx[i], self.shift_idx[i+1])
                left.append(self.images[l])
                right.append(self.images[r])
                sim.append([1])
                count += 1
            for j in range(nbr_each_fingerprint):
                # Generate random index of current digit i
                rnd_current_finger = random.randint(self.shift_idx[i], self.shift_idx[i+1])
                # Generate random index of digit not being i
                while True:
                    rnd_other_finger = random.randint(self.shift_idx[0], self.shift_idx[-1])
                    if not self.shift_idx[i] <= rnd_other_finger <= self.shift_idx[i+1]:
                        break
                      
                # Put the pair in random left/right
                if random.uniform(0,1) < 0.5:
                    left.append(self.images[rnd_current_finger])
                    right.append(self.images[rnd_other_finger])
                else:
                    left.append(self.images[rnd_other_finger])
                    right.append(self.images[rnd_current_finger])                
                    
                sim.append([0])
                count += 1
                
        # Generate remaining pairs in the batch
        # Make matching pairs every second time
        mat = 0
        while count < batch_size:
            new_finger = random.randint(0,len(self.shift_idx)-2)
            index_finger = random.randint(self.shift_idx[new_finger], self.shift_idx[new_finger+1])
            if mat == 0:    # Make unmatched pair
                while True:
                    index_non_match = random.randint(self.shift_idx[0], self.shift_idx[-1])
                    if not self.shift_idx[new_finger] <= index_non_match <= self.shift_idx[new_finger+1]:
                        break
                left.append(self.images[index_finger])
                right.append(self.images[index_non_match])
                sim.append([0])
                mat = 1 # Set next pair to be matching
            else:           # Make matching pair
                index_match = random.randint(self.shift_idx[new_finger], self.shift_idx[new_finger+1])
                left.append(self.images[index_finger])
                right.append(self.images[index_match])
                sim.append([1])
                mat = 0 #Set next pair to be non matching
            count += 1
                
        # Shuffle the data pairs and corresponding labels
        data_list = [left,right,sim]
        shuffled_data_list = util.shuffle_data(data_list)
            
        return np.array(shuffled_data_list[0]),np.array(shuffled_data_list[1]),shuffled_data_list[2]
    
    def prep_eval_data_pair(self, nbr_of_image_pairs):
        left = []
        right = []
        sim = []
        mat = 0
        count = 0
        while count < nbr_of_image_pairs:
            new_finger = random.randint(0,len(self.shift_idx)-2)
            index_finger = random.randint(self.shift_idx[new_finger], self.shift_idx[new_finger+1])
            if mat == 0:    # Make unmatched pair
                while True:
                    index_non_match = random.randint(self.shift_idx[0], self.shift_idx[-1])
                    if not self.shift_idx[new_finger] <= index_non_match <= self.shift_idx[new_finger+1]:
                        break
                left.append(self.images[index_finger])
                right.append(self.images[index_non_match])
                sim.append([0])
                mat = 1 # Set next pair to be matching
            else:           # Make matching pair
                index_match = random.randint(self.shift_idx[new_finger], self.shift_idx[new_finger+1])
                left.append(self.images[index_finger])
                right.append(self.images[index_match])
                sim.append([1])
                mat = 0 #Set next pair to be non matching
            count += 1
        
        return np.array(left),np.array(right),np.array(sim)
