#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 11:34:04 2018

@author: Tuong Lam & Simon Nilsson
"""

import imageio as imio
import numpy as np
import os
import sys

def fingerprint_parser(database_dir):
    """ Parser for CASIA fingerprint database.
    
    Input:
    database_dir - directory of the index file (ending with a forward slash "/")
    Returns:   
    """
    l_person_id = []
    l_finger_id = []
    l_fingerprints = []
    r_person_id = []
    r_finger_id = []
    r_fingerprints = []
            
    for dir in os.listdir(database_dir):
        for person_dir in os.listdir(database_dir + dir + "/"):
            for hand_dir in os.listdir(database_dir + dir + "/" + person_dir + "/"):
                curr_dir = database_dir + dir + "/" + person_dir + "/" + hand_dir + "/"
                for file in os.listdir(curr_dir):
                    if file == "Thumbs.db":
                        continue
                    words = file.split("_")
                    try:
                        person_id = int(words[0])
                        finger_id = int(words[1][1])
                    except ValueError:
                        print("Error for file: " + curr_dir + file)
                        continue
                    finger = imio.imread(curr_dir + file)
                    fingerprint = np.ndarray.flatten(np.array(finger))
                    if hand_dir == "L":
                        l_person_id.append(person_id)
                        l_finger_id.append(finger_id)
                        l_fingerprints.append(fingerprint)
                    else:
                        r_person_id.append(person_id)
                        r_finger_id.append(finger_id)
                        r_fingerprints.append(fingerprint)
                
    return l_person_id, l_finger_id, l_fingerprints, r_person_id, r_finger_id, r_fingerprints
    

def main(argv):
    database_dir_path = argv[0] # path to database
    dir_path = argv[1] # path to directory where the parsed data will be saved
    
    l_person_id, l_finger_id, l_fingerprints, r_person_id, r_finger_id, r_fingerprints = fingerprint_parser(database_dir_path)
    
    l_person_id = np.array(l_person_id, dtype="int32")
    l_finger_id = np.array(l_finger_id, dtype="int32")
    l_fingerprints = np.array(l_fingerprints, dtype ="float32")/255
    r_person_id = np.array(r_person_id, dtype="int32")
    r_finger_id = np.array(r_finger_id, dtype="int32")
    r_fingerprints = np.array(r_fingerprints, dtype ="float32")/255
    
    # save paths
    filename_1 = dir_path + "casia_l_person_id"
    filename_2 = dir_path + "casia_l_finger_id"
    filename_3 = dir_path + "casia_l_fingerprints" 
    filename_4 = dir_path + "casia_r_person_id"
    filename_5 = dir_path + "casia_r_finger_id"
    filename_6 = dir_path + "casia_r_fingerprints"
    
    # save numpy arrays
    np.save(filename_1, l_person_id)
    np.save(filename_2, l_finger_id)
    np.save(filename_3, l_fingerprints)
    np.save(filename_4, r_person_id)
    np.save(filename_5, r_finger_id)
    np.save(filename_6, r_fingerprints)
    
if __name__ == "__main__":
    main(sys.argv[1:])