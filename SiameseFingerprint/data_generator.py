# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:24:03 2018

@author: Tuong Lam
"""

import numpy as np
import random
import utilities as util

class data_generator:
    
    def __init__(self, images, finger_id, person_id, train_size, translation, rotation):
        self.finger_id = finger_id[0:train_size]
        self.person_id = person_id[0:train_size]
        self.images = images[0:train_size,:,:,:]
        self.translation = translation[0:train_size]
        self.rotation = rotation[0:train_size]
        self.shift_idx = [] # list with indices where fingerId changes
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
        
        # Simularity thresholds
        rot_sim = 5     # 5 degrees rotation threshold
        trans_sim = 30  # 30 pixels translation threshold
        # Generate matching pair indecies
        self.match = []
        self.no_match = []
        for i in range(len(self.shift_idx)-1):
            template_trans = translation[self.shift_idx[i]]
            template_rot = rotation[self.shift_idx[i]]
            for j in range(self.shift_idx[i]+1, self.shift_idx[i+1]):
                rot_match = False
                trans_match = False
                trans_cand = translation[j]
                rot_cand_interval = np.zeros(2)
                
                # Check if image is close in rotation to template
                if rotation[j] - rot_sim < 0:
                    rot_cand_interval[1] = rotation[j] + rot_sim
                    rot_cand_interval[0] = 360 - (rotation[j] - rot_sim)
                    if template_rot < rot_cand_interval[1] or template_rot > rot_cand_interval[0]:
                        rot_match = True
                elif rotation[j] + rot_sim > 360:
                    rot_cand_interval[1] = rotation[j] + rot_sim - 360
                    rot_cand_interval[0] = rotation[j] - rot_sim
                    if template_rot < rot_cand_interval[1] or template_rot > rot_cand_interval[0]:
                        rot_match = True
                else:
                    rot_cand_interval[1] = rotation[j] + rot_sim
                    rot_cand_interval[0] = rotation[j] - rot_sim
                    if template_rot < rot_cand_interval[1] or template_rot > rot_cand_interval[0]:
                        rot_match = True
                        
                # If rotation close check if translation is close
                if rot_match:
                    dx = np.abs(template_trans[0] - trans_cand[0])
                    dy = np.abs(template_trans[1] - trans_cand[1])
                    if dx < trans_sim and dy < trans_sim:
                        trans_match = True
                
                if trans_match and rot_match:
                    self.match.append([self.shift_idx[i], j])
                else:
                    self.no_match.append([self.shift_idx[i], j])

#    def __init__(self, images, finger_id, person_id, train_size):
#        self.finger_id = finger_id[0:train_size]
#        self.person_id = person_id[0:train_size]
#        self.images = images[0:train_size,:,:,:]
#        self.shift_idx = []
#        idx_counter = 0
#        nbr_of_persons = person_id[-1]
#        for i in range(nbr_of_persons):
#            finger_counter = 0
#            while idx_counter < len(person_id):
#                if not i+1 == self.person_id[idx_counter]:
#                    break
#                if not finger_counter == self.finger_id[idx_counter]:
#                    finger_counter = self.finger_id[idx_counter]
#                    self.shift_idx.append(idx_counter)
#                
#                idx_counter += 1
#                
#        self.shift_idx.append(len(self.person_id) - 1)

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
                    
                util.rand_assign_pair(left,right,self.images[rnd_current_finger],self.images[rnd_other_finger])
                sim.append([0])
                count += 1
                
        # Generate remaining pairs in the batch
        # Make matching pairs every second time
        mat = 0
        while count < batch_size:
            left_tmp,right_tmp,sim_tmp = util.generate_pair(self.images,self.shift_idx,mat)
            left.append(left_tmp)
            right.append(right_tmp)
            sim.append(sim_tmp)
            
            # if a dissimilar pair was generated, generate a matching pair next time and vice versa
            if sim_tmp[0] == 0: 
                mat = 1
            else:
                mat = 0
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
            left_tmp,right_tmp,sim_tmp = util.generate_pair(self.images,self.shift_idx,mat)
            left.append(left_tmp)
            right.append(right_tmp)
            sim.append(sim_tmp)
            
            # if a dissimilar pair was generated, generate a matching pair next time and vice versa
            if sim_tmp[0] == 0: 
                mat = 1
            else:
                mat = 0
            count += 1
        
        return np.array(left),np.array(right),np.array(sim)

    def gen_match_batch(self, batch_size):
        left = []
        right = []
        sim = []
        switch_match = 0
        for i in range(batch_size):
            if switch_match == 0:
                rnd_match = self.match[random.randint(0,len(self.match)-1)]
                util.rand_assign_pair(left,right,self.images[rnd_match[0]],self.images[rnd_match[1]])
                sim.append([1])
                switch_match = 1
            else:
                rnd_no_match = self.no_match[random.randint(0,len(self.no_match)-1)]
                util.rand_assign_pair(left,right,self.images[rnd_no_match[0]],self.images[rnd_no_match[1]])
                sim.append([0])
                switch_match = 0
        
        left, right, sim =  util.shuffle_data([left, right, sim])    
        
        return np.array(left), np.array(right), np.array(sim)
    
    def prep_eval_match(self, nbr_of_image_pairs):
        left = []
        right = []
        sim = []
        mat = 0
        count = 0
        while count < nbr_of_image_pairs:
            if mat == 0:    # Make unmatched pair
                rnd_no_match = random.randint(0,len(self.no_match)-1)
                left.append(self.images[self.no_match[rnd_no_match][0]])
                right.append(self.images[self.no_match[rnd_no_match][1]])
                sim.append([0])
                mat = 1 # Set next pair to be matching
            else:           # Make matching pair
                rnd_match = random.randint(0,len(self.match)-1)
                left.append(self.images[self.match[rnd_match][0]])
                right.append(self.images[self.match[rnd_match][1]])
                sim.append([1])
                mat = 0 #Set next pair to be non matching
            count += 1
        
        return np.array(left),np.array(right),np.array(sim)
