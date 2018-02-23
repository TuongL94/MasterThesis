# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:24:03 2018

@author: Tuong Lam
"""

import numpy as np
import random
import utilities as util
import math
import tensorflow as tf

class data_generator:
    
    def __init__(self, images, finger_id, person_id, translation, rotation, data_size, rotation_res):
        """ Initializes an instance of a data_generator class
        
        The data_generator contain attributes referring to the input data.
        The input data_size specifies the number of examples one wants to use
        from the original input data.
        Input:
        images - 4D numpy array of the format [nbr_of_images,height,width,1]
        finger_id - numpy array containing finger ids specified as integers (1,2,3,6,7,8)
        person_id - numpy array containing person ids specified as integers [0,inf)
        translation - 2D numpy array with rows corresponding to 2D translations
        rotation - numpy array containing rotation of images given in degrees
        data_size - amount of data one wants to use from original data 
        """
#        self.finger_id = finger_id[0:data_size]
#        self.person_id = person_id[0:data_size]
#        self.images = images[0:data_size,:,:,:]
#        self.translation = translation[0:data_size]
#        self.rotation = rotation[0:data_size]
#        self.breakpoints = [] # list with indices where fingerId changes

        # Split fingerprint data into training, validation and testing sets
        percentages = [0.8,0.1]
        self.train_data, self.val_data, self.test_data = self.three_split_array(images[0:data_size,:,:,:], percentages)
        self.train_finger_id, self.val_finger_id, self.test_finger_id = self.three_split_array(finger_id[0:data_size], percentages)
        self.train_person_id, self.val_person_id, self.test_person_id= self.three_split_array(person_id[0:data_size], percentages)
        self.train_translation, self.val_translation, self.test_translation= self.three_split_array(translation[0:data_size], percentages)
        self.train_rotation, self.val_rotation, self.test_rotation= self.three_split_array(rotation[0:data_size], percentages)
        

        # Breakpoints for training data 
        self.breakpoints_train = self.get_breakpoints(self.train_person_id, self.train_finger_id)
        
        # Breakpoints for validation data 
        self.breakpoints_val= self.get_breakpoints(self.val_person_id, self.val_finger_id)
        
        # Breakpoints for test data 
        self.breakpoints_test= self.get_breakpoints(self.test_person_id, self.test_finger_id)
        
        
        
#        idx_counter = 0
#        nbr_of_persons = self.person_id[-1]
#        for i in range(nbr_of_persons):
#            finger_counter = 0
#            while idx_counter < len(self.person_id):
#                if not i+1 == self.person_id[idx_counter]:
#                    break
#                if not finger_counter == self.finger_id[idx_counter]:
#                    finger_counter = self.finger_id[idx_counter]
#                    self.breakpoints.append(idx_counter)
#                
#                idx_counter += 1
#                
#        self.breakpoints.append(len(self.person_id) - 1)
        
#        self.match_train, self.no_match_train = self.gen_pair_indices(5,80,training = True)
#        self.match_eval, self.no_match_eval = self.gen_pair_indices(5,80,training = False)
        
        # All combinations of training data
        self.match_train, self.no_match_train = self.all_combinations(self.breakpoints_train, self.train_rotation, self.train_translation, 5, 80)
        
        # All combinations of training data
        self.match_val, self.no_match_val= self.all_combinations(self.breakpoints_val, self.val_rotation, self.val_translation, 5, 80)
        
        # All combinations of training data
        self.match_test, self.no_match_test= self.all_combinations(self.breakpoints_test, self.test_rotation, self.test_translation, 5, 80)
        
#        self.all_match, self.all_no_match = self.all_combinations(5, 80)
#        percentages = [0.8,0.1]
#        self.match_train, self.match_val, self.match_test = self.three_split_array(self.all_match,percentages)        
        # This will make us train on images in the test and validation set (we still get pairs that are not trained on)
#        self.no_match_train, self.no_match_val, self.no_match_test = self.three_split_array(self.all_no_match,percentages)
        
        # Create rotated training set with 90 degree steps
        original_train_data = self.train_data
        self.train_data = [self.train_data]
        self.rotation_res = rotation_res
        sess = tf.Session()
        with sess.as_default():
            for i in range(1,rotation_res):
                self.train_data.append(list(tf.contrib.image.rotate(np.array(original_train_data), math.pi/i).eval()))
    
    
    def get_breakpoints(self, person_id, finger_id):
        breakpoints = []
        idx_counter = 0
        nbr_of_persons = person_id[-1]
        for i in range(person_id[0], nbr_of_persons+1):
            finger_counter = 0
            while idx_counter < len(person_id):
                if not i == person_id[idx_counter]:
                    break
                if not finger_counter == finger_id[idx_counter]:
                    finger_counter = finger_id[idx_counter]
                    breakpoints.append(idx_counter)
                
                idx_counter += 1
                
        breakpoints.append(len(person_id) - 1)
        
        return breakpoints
    
    
    def get_pairs(self,data,pair_list):
        left_pairs = np.take(data,pair_list[:,0],axis=0)
        right_pairs = np.take(data,pair_list[:,1],axis=0)
        return left_pairs, right_pairs
    
    
    def is_rotation_similar(self,angle_1,angle_2,rotation_diff):
        """ Checks if two angles differ by at most rotation_diff in absolute value.
        
        Input:
        angle_1 - first angle, specified in degrees [0,360]
        angle_2 - second angle, specified in degrees [0,360]
        rotation_diff - difference in rotation
        Returns: True if the angles differ by at most rotation_diff in absolute value,
        otherwise False
        """
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
        """ Checks if two 2D translations correspond to points that are close to each other.
        
        The translatios are assumed to be relative to a fixed point in 2D space.
        Input:
        translation_1 - first translation, specified as a numpy array of two elements (x,y)
        translation_2 - second translation, specified as a numpy array of two elements (x,y)
        translation_diff - maximum distance between the translations in each axis
        Returns: True if the distance between the translations is at most translation_diff in each axis,
        otherwise False
        """
        translation_match = False
        dx = np.abs(translation_1[0] - translation_2[0])
        dy = np.abs(translation_1[1] - translation_2[1])
        
        if dx < translation_diff and dy < translation_diff:
            translation_match = True
            
        return translation_match
    
    def gen_pair_indices(self,rotation_diff, translation_diff, training = True):
        """ Generates indices for matching and non-matching pairs of images.
        
        This method generates indices for matching and non-matching pairs of images.
        If training is set to True it will generate indices used in training,
        otherwise indices for evaluation will be generated. Matching pairs are
        defined by images with sufficiently small rotation and translation
        difference, specified by rotation_diff and translation_diff.
        Input:
        rotation_diff - maximum allowed difference in rotation between two matching images
        translation_diff - maximum allowed distance between translations of two matching images
        in each axis
        training - parameter to determine if indices should be generated for training or evaluation.
        If training = True indices for training are generated otherwise the indices are for evaluation
        Returns:
        match - list of matching indices
        no_match - numpy array of non-matching indices, one row corresponds to one non-matching indices
        """
        match = [] # matching pair indices
        no_match = [] # non-matching pair indices 
        
        if training:
            for i in range(len(self.breakpoints)-1):
                template_trans = self.translation[self.breakpoints[i]]
                template_rot = self.rotation[self.breakpoints[i]]
                for j in range(self.breakpoints[i]+1, self.breakpoints[i+1]):
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
                        match.append([self.breakpoints[i], j])
                    else:
                        no_match.append([self.breakpoints[i], j])
        else:
            for i in range(len(self.breakpoints)-1):
                template_trans = self.translation[self.breakpoints[i]+1]
                template_rot = self.rotation[self.breakpoints[i]+1]
                for j in range(self.breakpoints[i]+2, self.breakpoints[i+1]):
                    rot_cand = self.rotation[j]
                    trans_cand = self.translation[j]
                    translation_match = False
                    rotation_match = self.is_rotation_similar(template_rot,rot_cand,rotation_diff)
                    
                    if rotation_match:
                        translation_match = self.is_translation_similar(template_trans,trans_cand,translation_diff)
                    
                    if translation_match and rotation_match:
                        match.append([self.breakpoints[i]+1, j])
                    else:
                        no_match.append([self.breakpoints[i]+1, j])
    
        # Shuffle no_match's columns so that it contains different non-matching fingers 
        no_match = np.array(no_match)
        np.random.shuffle(no_match)
        return match,no_match
    
    def all_combinations(self, breakpoints, rotation, translation, rotation_diff, translation_diff):
        match = [] # matching pair indices
        no_match = [] # non-matching pair indices 
        
        for i in range(len(breakpoints)-1):
            for k in range(breakpoints[i+1]-breakpoints[i]):
                template_trans = translation[breakpoints[i]+k]
                template_rot = rotation[breakpoints[i]+k]
                for j in range(breakpoints[i]+k+1, breakpoints[i+1]):
                    rot_cand = rotation[j]
                    trans_cand = translation[j]
                    translation_match = False
                    rotation_match = self.is_rotation_similar(template_rot,rot_cand,rotation_diff)
                    
                    # if rotation is sufficiently similar check translation
                    if rotation_match:
                        translation_match = self.is_translation_similar(template_trans,trans_cand,translation_diff)
                    
                    # if rotation and translation is similar the images related to the corresponding
                    # shift indices are considered similar
                    if translation_match and rotation_match:
                        match.append([breakpoints[i]+k, j])
                    else:
                        no_match.append([breakpoints[i]+k, j])
                    
                for n in range(breakpoints[i+1], rotation.shape[0]):
                    no_match.append([breakpoints[i]+k, n])
            
            
        no_match = np.array(no_match,dtype='int32')
        np.random.shuffle(no_match)
        return match,no_match
    
#    def all_combinations(self, rotation_diff, translation_diff):
#        match = [] # matching pair indices
#        no_match = [] # non-matching pair indices 
#        
#        for i in range(len(self.breakpoints)-1):
#            for k in range(self.breakpoints[i+1]-self.breakpoints[i]):
#                template_trans = self.translation[self.breakpoints[i]+k]
#                template_rot = self.rotation[self.breakpoints[i]+k]
#                for j in range(self.breakpoints[i]+k+1, self.breakpoints[i+1]):
#                    rot_cand = self.rotation[j]
#                    trans_cand = self.translation[j]
#                    translation_match = False
#                    rotation_match = self.is_rotation_similar(template_rot,rot_cand,rotation_diff)
#                    
#                    # if rotation is sufficiently similar check translation
#                    if rotation_match:
#                        translation_match = self.is_translation_similar(template_trans,trans_cand,translation_diff)
#                    
#                    # if rotation and translation is similar the images related to the corresponding
#                    # shift indices are considered similar
#                    if translation_match and rotation_match:
#                        match.append([self.breakpoints[i]+k, j])
#                    else:
#                        no_match.append([self.breakpoints[i]+k, j])
#                    
#                    for n in range(self.breakpoints[i+1], self.images.shape[0]):
#                        no_match.append([self.breakpoints[i]+k, n])
#            
#            
#        no_match = np.array(no_match)
#        np.random.shuffle(no_match)
#        return match,no_match

#    def __init__(self, images, finger_id, person_id, train_size):
#        self.finger_id = finger_id[0:train_size]
#        self.person_id = person_id[0:train_size]
#        self.images = images[0:train_size,:,:,:]
#        self.breakpoints = []
#        idx_counter = 0
#        nbr_of_persons = person_id[-1]
#        for i in range(nbr_of_persons):
#            finger_counter = 0
#            while idx_counter < len(person_id):
#                if not i+1 == self.person_id[idx_counter]:
#                    break
#                if not finger_counter == self.finger_id[idx_counter]:
#                    finger_counter = self.finger_id[idx_counter]
#                    self.breakpoints.append(idx_counter)
#                
#                idx_counter += 1
#                
#        self.breakpoints.append(len(self.person_id) - 1)

#    def gen_pair_batch(self,batch_size):
#        left = []
#        right = []
#        sim = []
#        count = 0
#        
#        nbr_fingerprints = len(self.finger_id) - 1
#        for i in range(nbr_fingerprints):
#            nbr_each_fingerprint = int(batch_size/nbr_fingerprints/2);
#            for j in range(nbr_each_fingerprint):
#                l = random.randint(self.breakpoints[i], self.breakpoints[i+1])
#                r = random.randint(self.breakpoints[i], self.breakpoints[i+1])
#                left.append(self.images[l])
#                right.append(self.images[r])
#                sim.append([1])
#                count += 1
#            for j in range(nbr_each_fingerprint):
#                # Generate random index of current digit i
#                rnd_current_finger = random.randint(self.breakpoints[i], self.breakpoints[i+1])
#                # Generate random index of digit not being i
#                while True:
#                    rnd_other_finger = random.randint(self.breakpoints[0], self.breakpoints[-1])
#                    if not self.breakpoints[i] <= rnd_other_finger <= self.breakpoints[i+1]:
#                        break
#                    
#                util.rand_assign_pair(left,right,self.images[rnd_current_finger],self.images[rnd_other_finger])
#                sim.append([0])
#                count += 1
#                
#        # Generate remaining pairs in the batch
#        # Make matching pairs every second time
#        mat = 0
#        while count < batch_size:
#            left_tmp,right_tmp,sim_tmp = util.generate_pair(self.images,self.breakpoints,mat)
#            left.append(left_tmp)
#            right.append(right_tmp)
#            sim.append(sim_tmp)
#            
#            # if a dissimilar pair was generated, generate a matching pair next time and vice versa
#            if sim_tmp[0] == 0: 
#                mat = 1
#            else:
#                mat = 0
#            count += 1
#                
#        # Shuffle the data pairs and corresponding labels
#        data_list = [left,right,sim]
#        shuffled_data_list = util.shuffle_data(data_list)
#            
#        return np.array(shuffled_data_list[0]),np.array(shuffled_data_list[1]),shuffled_data_list[2]
    
#    def prep_eval_data_pair(self, nbr_of_image_pairs):
#        left = []
#        right = []
#        sim = []
#        mat = 0
#        count = 0
#        while count < nbr_of_image_pairs:
#            left_tmp,right_tmp,sim_tmp = util.generate_pair(self.images,self.breakpoints,mat)
#            left.append(left_tmp)
#            right.append(right_tmp)
#            sim.append(sim_tmp)
#            
#            # if a dissimilar pair was generated, generate a matching pair next time and vice versa
#            if sim_tmp[0] == 0: 
#                mat = 1
#            else:
#                mat = 0
#            count += 1
#        
#        return np.array(left),np.array(right),np.array(sim)
#
    def gen_match_batch(self, batch_size):
        """ Generates a training batch with matched and non-matched pairs of images.
        
        Input:
        batch_size - the size of the batch
        Returns: three numpy arrays where the first two contain one image from randomly selected image pairs respectively.
        The last array indicates if corresponding pairs in the first two arrays are matching or non-matching
        pairs, if the pairs match the corresponding element in the last array is 1, otherwise 0.
        """
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
        
    
    def gen_batch(self, batch_size, training = 1):
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
        
        if training == 1:
            current_matching_set = self.match_train
            current_non_matching_set = self.no_match_train
        else:
            current_matching_set = self.match_val
            current_non_matching_set = self.no_match_val
            
        switch_match = 0
        for i in range(batch_size):
            if switch_match == 0:
                rnd_match = current_matching_set[random.randint(0,len(current_matching_set)-1)]
                util.rand_assign_pair(left,right,self.images[rnd_match[0]],self.images[rnd_match[1]])
                sim.append([1])
                switch_match = 1
            else:
                rnd_no_match = current_non_matching_set[random.randint(0,len(current_non_matching_set)-1)]
                util.rand_assign_pair(left,right,self.images[rnd_no_match[0]],self.images[rnd_no_match[1]])
                sim.append([0])
                switch_match = 0
        
        left, right, sim =  util.shuffle_data([left, right, sim])    
        
        return np.array(left), np.array(right), np.array(sim)
    
    def prep_eval_match(self, nbr_of_image_pairs):
        """ Generates an evaluation batch with matched and non-matched pairs of images.
        
        Input:
        nbr_of_image_pairs - the size of the evaluation batch
        Returns: three numpy arrays where the first two contain one image from randomly selected image pairs respectively.
        The last array indicates if corresponding pairs in the first two arrays are matching or non-matching
        pairs, if the pairs match the corresponding element in the last array is 1, otherwise 0.
        """
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

    def three_split_array(self,input_array,percentage):
        length = len(input_array)
        split_ind = [math.floor(length*percentage[0]), math.floor(length*percentage[0])+math.floor(length*percentage[1])]
        
        first_split = input_array[0:split_ind[0]+1]
        second_split = input_array[split_ind[0]+1:split_ind[1]+1]
        third_split = input_array[split_ind[1]+1:]
        return first_split,second_split,third_split
    
    
    def gen_seed0(self, counter, data_size, batch_size):
        if counter == 0:
            self.left_all, self.right_all, self.sim_all = self.gen_match_batch(data_size)
            
        idx_counter = counter % batch_size
        
        left = self.left_all[idx_counter:idx_counter+batch_size]
        right = self.right_all[idx_counter:idx_counter+batch_size]
        sim = self.sim_all[idx_counter:idx_counter+batch_size]
        
        return left,right,sim,counter+batch_size
            
    