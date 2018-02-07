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
import math

def fingerprint_parser(index_file_dir, index_file_name):
    person_id = []
    finger_id = []
    fingerprints = []
    translation = []
    rotation = []
    counter = 0

    with open(index_file_dir + index_file_name,"r") as file:
        for line in file:
            words = re.split("\t",line)
#            if words[0] != "#" and words[0] != "##":
            if len(words) > 4 and words[0] != "#":
                if len(words[4]) > 40:
                    last_word = re.split(":",words[-1])
                    alignment_word = last_word[-1].split()
                    person_id.append(int(words[0]))
                    finger_id.append(int(words[1]))
                    fingerprint_path = last_word[0].strip()
                    if counter % 100 == 0:
                        print(counter)
                    finger = imio.imread(index_file_dir + fingerprint_path)
                    fingerprints.append(np.ndarray.flatten(np.array(finger)))
                    translation.append([int(alignment_word[1]),int(alignment_word[2])])
                    rotation.append(int(alignment_word[3]))
                    counter += 1
                
    return person_id, finger_id, fingerprints, translation, rotation
    
def main(argv):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    person_id, finger_id, fingerprints, translation, rotation = fingerprint_parser(argv[0],argv[1])
    person_id = np.array(person_id)
    finger_id = np.array(finger_id)
    fingerprints = np.array(fingerprints)/255
    translation = np.array(translation)/256
    rotation = np.array(rotation)/65536*360 
        

    filename_1 = dir_path + "/person_id"
    filename_2 = dir_path + "/finger_id"
    filename_3 = dir_path + "/fingerprints" 
    filename_4 = dir_path + "/translation" 
    filename_5 = dir_path + "/rotation" 
    np.save(filename_1,person_id)
    np.save(filename_2,finger_id)
    np.save(filename_3,fingerprints)
    np.save(filename_4, translation)
    np.save(filename_5, rotation)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
