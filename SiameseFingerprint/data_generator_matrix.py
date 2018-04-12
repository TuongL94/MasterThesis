# -*- coding: utf-8 -*-
"""
Created on Tue Apr 9 9:11:03 2018

@author: Simon Nilsson
"""

import numpy as np
import math
import tensorflow as tf

import time

class data_generator_matrix:
    
    def __init__(self, images, finger_id, person_id, matching_matrix, class_breakpoints, data_size, rotation_res):
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

        # Split fingerprint data into training, validation and testing sets
        percentages = [0.8,0.1]
        self.train_data, self.val_data, self.test_data = self.three_split_array(images[0:data_size,:,:,:], percentages)
        self.train_finger_id, self.val_finger_id, self.test_finger_id = self.three_split_array(finger_id[0:data_size], percentages)
        self.train_person_id, self.val_person_id, self.test_person_id= self.three_split_array(person_id[0:data_size], percentages)
        
        self.train_matrix, self.val_matrix, self.test_matrix = self.three_split_array(matching_matrix[0:data_size], percentages)
        
        self.train_class_breakpoints, self.val_class_breakpoints, self.test_class_breakpoints = self.three_split_array(class_breakpoints[0:data_size], percentages)
        # Fix val and test breakpoints to start at 0
        self.test_class_breakpoints = self.test_class_breakpoints - self.test_class_breakpoints[0]
        self.val_class_breakpoints = self.val_class_breakpoints - self.val_class_breakpoints[0]
        
        # Breakpoints for training data 
        self.breakpoints_train = self.get_breakpoints(self.train_person_id, self.train_finger_id)
        
        # Breakpoints for validation data 
        self.breakpoints_val= self.get_breakpoints(self.val_person_id, self.val_finger_id)
        
        # Breakpoints for test data 
        self.breakpoints_test= self.get_breakpoints(self.test_person_id, self.test_finger_id)
        
        rot_diff = 5
        trans_diff = 10
        
#        # All combinations of training data
#        self.match_train, self.no_match_train = self.all_combinations(self.breakpoints_train, self.train_rotation, self.train_translation, rot_diff, trans_diff)
#        
#        # All combinations of training data
#        self.match_val, self.no_match_val= self.all_combinations(self.breakpoints_val, self.val_rotation, self.val_translation, rot_diff, trans_diff)
#        
#        # All combinations of training data
#        self.match_test, self.no_match_test= self.all_combinations(self.breakpoints_test, self.test_rotation, self.test_translation, rot_diff, trans_diff)
        
#        '''Make easy matching and non matching sets'''
#        margin_trans = 192
#        margin_rot = 20
#        # All combinations of training data
#        self.match_train, self.no_match_train = self.all_combinations_easy(self.breakpoints_train, self.train_rotation, self.train_translation, rot_diff, trans_diff, margin_rot, margin_trans)
#        
#        # All combinations of training data
#        self.match_val, self.no_match_val= self.all_combinations_easy(self.breakpoints_val, self.val_rotation, self.val_translation, rot_diff, trans_diff, margin_rot, margin_trans)
#        
#        # All combinations of training data
#        self.match_test, self.no_match_test= self.all_combinations_easy(self.breakpoints_test, self.test_rotation, self.test_translation, rot_diff, trans_diff, margin_rot, margin_trans)
        
        #############
        match_threshold = 80
        no_match_threshold = 10
        
        self.match_train, self.no_match_train = self.matrix_combinations(self.train_matrix, self.train_class_breakpoints, match_threshold, no_match_threshold)
        
        # All combinations of training data
        self.match_val, self.no_match_val = self.matrix_combinations(self.val_matrix, self.val_class_breakpoints, match_threshold, no_match_threshold)
        
        # All combinations of training data
        self.match_test, self.no_match_test = self.matrix_combinations(self.test_matrix, self.test_class_breakpoints, match_threshold, no_match_threshold)
        
        
        self.rotation_res = rotation_res
        self.gen_rotations(self.train_data,self.val_data,self.test_data,self.rotation_res)
        
    
    def add_new_data(self, images, finger_id, person_id, translation, rotation, data_size):
        percentages = [0.8,0.1]
        train_data, val_data, test_data = self.three_split_array(images[0:data_size,:,:,:], percentages)
        train_finger_id, val_finger_id, test_finger_id = self.three_split_array(finger_id[0:data_size], percentages)
        train_person_id, val_person_id, test_person_id= self.three_split_array(person_id[0:data_size], percentages)
        train_translation, val_translation, test_translation= self.three_split_array(translation[0:data_size], percentages)
        train_rotation, val_rotation, test_rotation= self.three_split_array(rotation[0:data_size], percentages)
        
        nbr_of_train_images = np.shape(self.train_data[0])[0]
        nbr_of_val_images = np.shape(self.val_data[0])[0]
        nbr_of_test_images = np.shape(self.test_data[0])[0]
        
        # Add new data to the current generator
        self.train_data[0] = np.append(self.train_data[0],train_data,axis=0)
        self.train_finger_id = np.append(self.train_finger_id,train_finger_id)
        self.train_person_id = np.append(self.train_person_id,train_person_id)
        self.train_rotation = np.append(self.train_rotation,train_rotation)
        self.train_translation = np.append(self.train_translation,train_translation,axis=0)
        self.val_data[0] = np.append(self.val_data[0],val_data,axis=0)
        self.val_finger_id = np.append(self.val_finger_id,val_finger_id)
        self.val_person_id = np.append(self.val_person_id,val_person_id)
        self.val_translation = np.append(self.val_translation,val_translation,axis=0)
        self.val_rotation = np.append(self.val_rotation,val_rotation)
        self.test_data[0] = np.append(self.test_data[0],test_data,axis=0)
        self.test_finger_id = np.append(self.test_finger_id,test_finger_id)
        self.test_person_id = np.append(self.test_person_id,test_person_id)
        self.test_rotation = np.append(self.test_rotation,test_rotation)
        self.test_translation = np.append(self.test_translation,test_translation,axis=0)
        
        self.gen_rotations(train_data,val_data,test_data,self.rotation_res) 
        
         # Breakpoints for training data 
        breakpoints_train = [e + nbr_of_train_images for e in self.get_breakpoints(train_person_id, train_finger_id)]
        
        # Breakpoints for validation data 
        breakpoints_val= [e + nbr_of_val_images for e in self.get_breakpoints(val_person_id, val_finger_id)]
        
        # Breakpoints for test data 
        breakpoints_test= [e + nbr_of_test_images for e in self.get_breakpoints(test_person_id, test_finger_id)]
    
        rot_diff = 5
        trans_diff = 10
        margin_trans = 192
        margin_rot = 20
        # All combinations of training data
        match_train, no_match_train = self.all_combinations_easy(breakpoints_train, self.train_rotation, self.train_translation, rot_diff, trans_diff, margin_rot, margin_trans)
        
        # All combinations of training data
        match_val, no_match_val= self.all_combinations_easy(breakpoints_val, self.val_rotation, self.val_translation, rot_diff, trans_diff, margin_rot, margin_trans)
        
        # All combinations of training data
        match_test, no_match_test= self.all_combinations_easy(breakpoints_test, self.test_rotation, self.test_translation, rot_diff, trans_diff, margin_rot, margin_trans)
       
        # Add new pair indices to the pair indices of the current generator
        self.match_train = np.append(self.match_train,match_train,axis=0)
        self.no_match_train = np.append(self.no_match_train,no_match_train,axis=0)
        self.match_val = np.append(self.match_val,match_val,axis=0)
        self.no_match_val = np.append(self.no_match_val,no_match_val,axis=0)
        self.match_test = np.append(self.match_test,match_test,axis=0)
        self.no_match_test = np.append(self.no_match_test,no_match_test,axis=0)
        
    def gen_rotations(self,train_data,val_data,test_data,rotation_res):
        original_train_data = train_data
        original_val_data = val_data
        original_test_data = test_data
        no_rotations_exist = type(self.train_data) is np.ndarray
        
        if no_rotations_exist:
            self.train_data = [self.train_data]
            self.val_data = [self.val_data]
            self.test_data = [self.test_data]

        train_dims = np.shape(original_train_data)
        val_dims = np.shape(original_val_data)
        test_dims = np.shape(original_test_data)
        train_holder = tf.placeholder(tf.float32,shape=[train_dims[0],train_dims[1],train_dims[2],train_dims[3]])
        val_holder = tf.placeholder(tf.float32,shape=[val_dims[0],val_dims[1],val_dims[2],val_dims[3]])
        test_holder = tf.placeholder(tf.float32,shape=[test_dims[0],test_dims[1],test_dims[2],test_dims[3]])
        
        with tf.Session() as sess:
            if no_rotations_exist:
                for i in range(1,rotation_res):
                    angle = 2*math.pi/rotation_res*i
                    rotated_train_images = sess.run(tf.contrib.image.rotate(train_holder,angle), feed_dict={train_holder:original_train_data})
                    rotated_val_images = sess.run(tf.contrib.image.rotate(val_holder,angle), feed_dict={val_holder:original_val_data})
                    rotated_test_images = sess.run(tf.contrib.image.rotate(test_holder,angle), feed_dict={test_holder:original_test_data})
                    self.train_data.append(rotated_train_images)
                    self.val_data.append(rotated_val_images)
                    self.test_data.append(rotated_test_images)
            else:
                for i in range(1,rotation_res):
                    angle = 2*math.pi/rotation_res*i
                    rotated_train_images = sess.run(tf.contrib.image.rotate(train_holder,angle), feed_dict={train_holder:original_train_data})
                    rotated_val_images = sess.run(tf.contrib.image.rotate(val_holder,angle), feed_dict={val_holder:original_val_data})
                    rotated_test_images = sess.run(tf.contrib.image.rotate(test_holder,angle), feed_dict={test_holder:original_test_data})
                    self.train_data[i] = np.append(self.train_data[i],rotated_train_images,axis=0)
                    self.val_data[i] = np.append(self.val_data[i],rotated_val_images,axis=0)
                    self.test_data[i] = np.append(self.test_data[i],rotated_test_images,axis=0)
                          
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
                
        breakpoints.append(len(person_id))
        
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
                    # breakpoint indices are considered similar
                    if translation_match and rotation_match:
                        match.append([breakpoints[i]+k, j])
                    else:
                        no_match.append([breakpoints[i]+k, j])
                    
                for n in range(breakpoints[i+1], rotation.shape[0]):    
                    no_match.append([breakpoints[i]+k, n])
            
        return np.array(match,dtype="int32"),np.array(no_match,dtype="int32")
    
    def all_combinations_easy(self, breakpoints, rotation, translation, rotation_diff, translation_diff, margin_rot, margin_trans):
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
                    translation_margin = False
                    rotation_match = self.is_rotation_similar(template_rot,rot_cand,rotation_diff)
#                    rotation_margin = self.is_rotation_similar(template_rot,rot_cand,margin_rot)
                    
                    # if rotation is sufficiently similar check translation
                    if rotation_match:
                        translation_match = self.is_translation_similar(template_trans,trans_cand,translation_diff)
#                    elif rotation_margin:
#                        translation_margin = self.is_translation_similar(template_trans,trans_cand,margin_trans)
                    translation_margin = self.is_translation_similar(template_trans,trans_cand,margin_trans)
                    
                    # if rotation and translation is similar the images related to the corresponding
                    # breakpoint indices are considered similar
                    if translation_match and rotation_match:
                        match.append([breakpoints[i]+k, j])
#                    elif rotation_margin and translation_margin:
#                        continue
                    elif translation_margin:
                        continue
                    else:
                        no_match.append([breakpoints[i]+k, j])
                    
                for n in range(breakpoints[i+1], rotation.shape[0]):        # Look strange, wrong???? Shouldn't it be breakpoints[i-1]???
                    no_match.append([breakpoints[i]+k, n])
            
        return np.array(match,dtype="int32"),np.array(no_match,dtype="int32")
    
            
    
    def matrix_combinations(self, matching_matrix, class_breakpoints, matching_threshold, non_matching_threshold):
        match = []
        no_match = []
        
        nbr_non_matching_other_class = 50
#        matching_matrix = list(matching_matrix)
        
        for i in range(len(class_breakpoints) - 1):
            current_matrix = matching_matrix[i]
#            for k in range(class_breakpoints[i+1] - class_breakpoints[i]):
            for k in range(current_matrix.shape[0]):
                for j in range(1, class_breakpoints[i+1] - class_breakpoints[i] - k):
                    if current_matrix[k,k + j] > matching_threshold:
                        match.append([class_breakpoints[i] + k, class_breakpoints[i] + k + j])
#                    elif current_matrix[k,k + j] < non_matching_threshold:
#                        no_match.append([class_breakpoints[i] + k, class_breakpoints[k] + k + j])

#                if i-2 > 0:
#                    for j in range(class_breakpoints[i-2]):
#                        no_match.append([class_breakpoints[i]+k, j])
                 
                nbr_non_matching_previous = int(i / len(class_breakpoints) * nbr_non_matching_other_class)
                nbr_non_matching_coming = int((1 - i / len(class_breakpoints)) * nbr_non_matching_other_class)
                
                if i-2 > 0 and i < len(class_breakpoints) - 2:
                    previous = np.random.randint(class_breakpoints[i-2], size=nbr_non_matching_previous)
                    coming = np.random.randint(class_breakpoints[i+1], class_breakpoints[-1], size=nbr_non_matching_coming)
                    for j in range(len(previous)):
                        no_match.append([class_breakpoints[i] + k, previous[j]])
                    for j in range(len(coming)):
                        no_match.append([class_breakpoints[i] + k, coming[j]])
                    
        return np.array(match, dtype="int32"), np.array(no_match, dtype="int32")

    
    def three_split_array(self,input_array,percentage):
        length = len(input_array)
        split_ind = [math.floor(length*percentage[0]), math.floor(length*percentage[0])+math.floor(length*percentage[1])]
        
        first_split = input_array[0:split_ind[0]+1]
        second_split = input_array[split_ind[0]+1:split_ind[1]+1]
        third_split = input_array[split_ind[1]+1:]
        return first_split,second_split,third_split
    
    def same_class(self, pairs, test = False):
        """Finds which pairs belongs to the same class (same finger and person i.e. same fingerprint)
        Input:
            pairs - (N x 2) matrix with pairs on each row
            test - optinal boolean to be set if the class test is to be run on the test set
        Return:
            class_id - (N x 1) vector with 0 and 1. 1 corresponds to pairs within the same class
        """
        nbr_of_pairs = pairs.shape[0]
        class_id = np.zeros((nbr_of_pairs, 1))
        
        if test:
            person_id = self.test_person_id
            finger_id = self.test_finger_id
        else:
            person_id = self.val_person_id
            finger_id = self.val_finger_id
     
        for i in range(nbr_of_pairs):
            first = [person_id[pairs[i,0]], finger_id[pairs[i,0]]]
            second = [person_id[pairs[i,1]], finger_id[pairs[i,1]]]
            if first == second:
                class_id[i] = 1    
            
        return class_id
        