# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:24:03 2018

@author: Tuong Lam
"""

import numpy as np
import random
import utilities as util

class data_generator:
    
    def __init__(self, images, finger_id, person_id, translation, rotation, data_size):
        self.finger_id = finger_id[0:data_size]
        self.person_id = person_id[0:data_size]
        self.images = images[0:data_size,:,:,:]
        self.translation = translation[0:data_size]
        self.rotation = rotation[0:data_size]
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
        
        self.match_train, self.no_match_train = self.gen_pair_indices(5,30,training = True)
        self.match_eval, self.no_match_eval = self.gen_pair_indices(5,30,training = False)
        self.all_match, self.all_no_match = self.all_combinations(5, 80)
        mat = self.all_match
        no = self.all_no_match
        
    def is_rotation_similar(self,angle_1,angle_2,rotation_diff):
        rot_cand_interval = np.zeros(2)
        rot_match = False
        if angle_2 - rotation_diff < 0:
            rot_cand_interval[1] = angle_2 + rotation_diff
            rot_cand_interval[0] = 360 - (angle_2 - rotation_diff)
            if angle_1 < rot_cand_interval[1] or angle_1 > rot_cand_interval[0]:
                rot_match = True
        elif angle_2 + rotation_diff > 360:
            rot_cand_interval[1] = angle_2 + rotation_diff - 360
            rot_cand_interval[0] = angle_2 - rotation_diff
            if angle_1 < rot_cand_interval[1] or angle_1 > rot_cand_interval[0]:
                rot_match = True
        else:
            rot_cand_interval[1] = angle_2 + rotation_diff
            rot_cand_interval[0] = angle_2 - rotation_diff
            if angle_1 < rot_cand_interval[1] or angle_1 > rot_cand_interval[0]:
                rot_match = True
                
        return rot_match
    
    def is_translation_similar(self,translation_1,translation_2, translation_diff):
        translation_match = False
        dx = np.abs(translation_1[0] - translation_2[0])
        dy = np.abs(translation_1[1] - translation_2[1])
        
        if dx < translation_diff and dy < translation_diff:
            translation_match = True
            
        return translation_match
    
    def gen_pair_indices(self,rotation_diff, translation_diff, training = True):
        match = [] # matching pair indices
        no_match = [] # non-matching pair indices 
        
        if training:
            for i in range(len(self.shift_idx)-1):
                template_trans = self.translation[self.shift_idx[i]]
                template_rot = self.rotation[self.shift_idx[i]]
                for j in range(self.shift_idx[i]+1, self.shift_idx[i+1]):
                    rot_cand = self.rotation[j]
                    trans_cand = self.translation[j]
                    translation_match = False
                    rotation_match = self.is_rotation_similar(template_rot,rot_cand,rotation_diff)
                    
                    # if rotation is sufficiently similar check translation
                    if rotation_match:
                        translation_match = self.is_translation_similar(template_trans,trans_cand,translation_diff)
                    
                    # if rotation and translation is similar the images related to the corresponding
                    # shift indices are considered similar
                    if translation_match and rotation_match:
                        match.append([self.shift_idx[i], j])
                    else:
                        no_match.append([self.shift_idx[i], j])
        else:
            for i in range(len(self.shift_idx)-1):
                template_trans = self.translation[self.shift_idx[i]+1]
                template_rot = self.rotation[self.shift_idx[i]+1]
                for j in range(self.shift_idx[i]+2, self.shift_idx[i+1]):
                    rot_cand = self.rotation[j]
                    trans_cand = self.translation[j]
                    translation_match = False
                    rotation_match = self.is_rotation_similar(template_rot,rot_cand,rotation_diff)
                    
                    if rotation_match:
                        translation_match = self.is_translation_similar(template_trans,trans_cand,translation_diff)
                    
                    if translation_match and rotation_match:
                        match.append([self.shift_idx[i]+1, j])
                    else:
                        no_match.append([self.shift_idx[i]+1, j])
    
        # Shuffle no_match's columns so that it contains different non-matching fingers 
        no_match = np.array(no_match)
        np.random.shuffle(no_match)
        return match,no_match
    
    def all_combinations(self, rotation_diff, translation_diff):
        match = [] # matching pair indices
        no_match = [] # non-matching pair indices 
        
       
        for i in range(len(self.shift_idx)-1):
            for k in range(self.shift_idx[i+1]-self.shift_idx[i]):
                template_trans = self.translation[self.shift_idx[i]+k]
                template_rot = self.rotation[self.shift_idx[i]+k]
#                k = 2
                for j in range(self.shift_idx[i]+k+1, self.shift_idx[i+1]):
                    rot_cand = self.rotation[j]
                    trans_cand = self.translation[j]
                    translation_match = False
                    rotation_match = self.is_rotation_similar(template_rot,rot_cand,rotation_diff)
                    
                    # if rotation is sufficiently similar check translation
                    if rotation_match:
                        translation_match = self.is_translation_similar(template_trans,trans_cand,translation_diff)
                    
                    # if rotation and translation is similar the images related to the corresponding
                    # shift indices are considered similar
                    if translation_match and rotation_match:
                        match.append([self.shift_idx[i]+k, j])
                    else:
                        no_match.append([self.shift_idx[i]+k, j])
#                    k += 1    
                    
            for n in range(self.shift_idx[i+1], self.images.shape[0]):
                no_match.append([self.shift_idx[i]+i, n])
            
            
        no_match = np.array(no_match)
        np.random.shuffle(no_match)
        return match,no_match

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
                rnd_match = self.match_train[random.randint(0,len(self.match_train)-1)]
                util.rand_assign_pair(left,right,self.images[rnd_match[0]],self.images[rnd_match[1]])
                sim.append([1])
                switch_match = 1
            else:
                rnd_no_match = self.no_match_train[random.randint(0,len(self.no_match_train)-1)]
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
                rnd_no_match = random.randint(0,len(self.no_match_eval)-1)
                left.append(self.images[self.no_match_eval[rnd_no_match][0]])
                right.append(self.images[self.no_match_eval[rnd_no_match][1]])
                sim.append([0])
                mat = 1 # Set next pair to be matching
            else:           # Make matching pair
                rnd_match = random.randint(0,len(self.match_eval)-1)
                left.append(self.images[self.match_eval[rnd_match][0]])
                right.append(self.images[self.match_eval[rnd_match][1]])
                sim.append([1])
                mat = 0 #Set next pair to be non matching
            count += 1
        
        return np.array(left),np.array(right),np.array(sim)
