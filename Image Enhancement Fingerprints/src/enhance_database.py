# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016

@author: utkarsh
"""

import numpy as np
#import cv2
#import numpy as np;
import matplotlib.pylab as plt;
import scipy.ndimage
import sys
from PIL import Image
from PIL import ImageEnhance
import os
 
from image_enhance import image_enhance


def adjust_contrast(input_image, output_image, factor):
#    image = Image.open(input_image)
    image = Image.fromarray(input_image)
    image = image.convert('L')
    enhancer_object = ImageEnhance.Contrast(image)
    out = enhancer_object.enhance(factor)
#    out.save(output_image)
    return out


def enhance_database(image_path, output_path, contrast_factor):
    
#    database_path = '/home/PRECISE/exjobb/Documents/MasterThesis/PB_database/CNN/PBFPC1020/PNG_192x192_v8_db'
    
#    person_id, finger_id, fingerprints, translation, rotation = fingerprint_parser(index_file_dir,index_file_name)
    
    try:
        img = scipy.ndimage.imread(image_path)
    except OSError as ex:
        print('Could not find any file named:  ' + image_path)
        return
        
#    if(len(img.shape)>2):
#        # img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
#        img = np.dot(img[...,:3], [0.299, 0.587, 0.114]);
    
    rows,cols = np.shape(img);
    aspect_ratio = np.double(rows)/np.double(cols);
    
    new_rows = 150;             # randomly selected number
    new_cols = new_rows/aspect_ratio;
    
    #img = cv2.resize(img,(new_rows,new_cols));
    img = scipy.misc.imresize(img,(np.int(new_rows),np.int(new_cols)));
    
    enhanced_img = image_enhance(img) + 128;    
    
#    contrast = 5
    enhanced_img_contrast = adjust_contrast(enhanced_img, "contrast", contrast_factor)
#    enhanced_img_contrast = np.array(enhanced_img_contrast)
    crop_margin = 15
    enhanced_img_contrast = enhanced_img_contrast.crop((crop_margin, crop_margin, new_rows-crop_margin, new_rows-crop_margin))
    
    # Save
    split_path = image_path.split('/')
    save_path = output_path + '/' + split_path[-2] + '/' + split_path[-1] 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scipy.misc.imsave(save_path, enhanced_img_contrast)
    

#image_path = '/home/PRECISE/exjobb/Documents/MasterThesis/PB_database/CNN/PBFPC1020/PNG_192x192_v8_db/000000001/._GT_000000001-01-000.png'

database_path = '/home/PRECISE/exjobb/Documents/MasterThesis/PB_database/CNN/PBFPC1020/PNG_192x192_v8_db'
output_path = '/home/PRECISE/exjobb/Documents/MasterThesis/PB_database/GaborFiltered_192x192'
contrast_factor = 5
count_persons = -1
for subdir, dirs, files in os.walk(database_path):
    current_person = subdir.split('/')[-1]
    print('Starting to enhance images from: ' +  current_person)
    print('Done %d person out of %d' % (count_persons, 156))
    for file in files:
        if file[-4:] == '.png':
            enhance_database(os.path.join(subdir, file), output_path, contrast_factor)
    count_persons += 1
    
    
