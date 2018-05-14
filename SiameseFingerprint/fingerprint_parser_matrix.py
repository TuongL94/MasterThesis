# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:31:04 2018

@author: Tuong Lam & Simon Nilsson
"""

import imageio as imio
import numpy as np
import re
import os
import sys
import pickle


def fingerprint_parser_matrix(index_file_dir, index_file_name):
    """ Parser for Precise Biometrics fingerprint database with alignment data.
    
    Input:
    index_file_dir - directory of the index file (ending with a forward slash "/")
    index_file_name - name of the index file
    Returns: lists with information used in the siamese network for fingerprint verification    
    """
    person_id = []
    finger_id = []
    fingerprints = []
    translation = []
    rotation = []
    matching_matrices = []
    fingers_in_each_class = []
    nbr_fingers = 0
    counter = 0
    last_finger_id = 0
    class_breakpoint = [0]
    interclass_counter = 0

    with open(index_file_dir + index_file_name,"r") as file:
        for line in file:
            words = re.split("\t",line)
            if len(words) > 4 and words[0] != "#":
                if len(words[4]) > 40: # only consider data that contains alignment information 
                    last_word = re.split(":",words[-1])
                    alignment_word = last_word[-1].split()
                    person_id.append(int(words[0]))
                    finger_id.append(int(words[1]))
                    fingerprint_path = last_word[0].strip()
#                    if counter % 100 == 0:
#                        print(counter)
                    finger = imio.imread(index_file_dir + fingerprint_path)
                    fingerprints.append(np.ndarray.flatten(np.array(finger)))
                    translation.append([int(alignment_word[1]),int(alignment_word[2])])
                    rotation.append(int(alignment_word[3]))
                    
                    # matrix entry for current fingerprint
                    matching = [int(s) for s in last_word[1].split()]
                    matching = np.array(matching[4:])
                    if last_finger_id == finger_id[-1] and nbr_fingers < 45:
                        matrix = np.vstack((matrix,matching))
                        nbr_fingers += 1
                        interclass_counter += 1
                    else:
                        if counter == 0:
                            matrix = matching
                        else:
                            fingers_in_each_class.append(nbr_fingers)
                            nbr_fingers = 0
                            class_breakpoint.append(counter)
                            interclass_counter = 0
                            matrix = matrix[:,:matrix.shape[0]]
                            matching_matrices.append(matrix)
                            matrix = matching
                            
                    last_finger_id = finger_id[-1]    
                    
                    counter += 1
                    
    class_breakpoint.append(counter)
    fingers_in_each_class.append(nbr_fingers)
    matrix = matrix[:,:matrix.shape[0]]
    matching_matrices.append(matrix)
                    
    return person_id, finger_id, fingerprints, matching_matrices, fingers_in_each_class, class_breakpoint


def main(argv):
#    dir_path = os.path.dirname(os.path.realpath(__file__)) # directory of file being executed
    dir_path = argv[2]
    person_id, finger_id, fingerprints, matrix, nbr_in_class, class_breakpoint = fingerprint_parser_matrix(argv[0],argv[1])
    
    # convert to numpy arrays and corrects scaling 
    person_id = np.array(person_id,dtype='int32')
    finger_id = np.array(finger_id,dtype='int32')
    fingerprints = np.array(fingerprints,dtype='float32')/255

    # save paths
    filename_1 = dir_path + "person_id_mt_vt_112"
    filename_2 = dir_path + "finger_id_mt_vt_112"
    filename_3 = dir_path + "fingerprints_mt_vt_112" 
    filename_4 = dir_path + "matrix_mt_vt_112"
    filename_5 = dir_path + "nbr_in_class_mt_vt_112"
    filename_6 = dir_path + "class_breakpoint_mt_vt_112"
    
    # saves numpy arrays
    np.save(filename_1, person_id)
    np.save(filename_2, finger_id)
    np.save(filename_3, fingerprints)
    np.save(filename_4, matrix)
    np.save(filename_5, np.array(nbr_in_class))
    np.save(filename_6, np.array(class_breakpoint))
    
if __name__ == "__main__":
    main(sys.argv[1:])
