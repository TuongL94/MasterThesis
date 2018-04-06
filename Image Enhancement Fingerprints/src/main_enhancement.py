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


if(len(sys.argv)<2):
    print('loading sample image');
    img_name = 'GT2.png'
    try:
#        img = scipy.ndimage.imread('../images/' + img_name);
        img = scipy.ndimage.imread('/home/PRECISE/exjobb/Documents/MasterThesis/PB_database/CNN/PBFPC1020/PNG_192x192_v8_db/000000001/._GT_000000001-01-000.png')
    except OSError as ex:
        print('Image file does not exist')
        
elif(len(sys.argv) >= 2):
    img_name = sys.argv[1];
    img = scipy.ndimage.imread(sys.argv[1]);
    
if(len(img.shape)>2):
    # img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114]);

#subdir = '/home/PRECISE/exjobb/Documents/MasterThesis/PB_database/CNN/PBFPC1020/PNG_192x192_v8_db/000000156'
#file = 'GT_000000156-08-10.png'
#img = scipy.ndimage.imread(os.path.join(subdir, file))

rows,cols = np.shape(img);
aspect_ratio = np.double(rows)/np.double(cols);

new_rows = 150;             # randomly selected number
new_cols = new_rows/aspect_ratio;

#img = cv2.resize(img,(new_rows,new_cols));
img = scipy.misc.imresize(img,(np.int(new_rows),np.int(new_cols)));

enhanced_img = image_enhance(img) + 128;   

contrast = 5
enhanced_img_contrast = adjust_contrast(enhanced_img, "contrast", contrast)
crop_margin = 15
enhanced_img_contrast = enhanced_img_contrast.crop((crop_margin, crop_margin, new_rows-crop_margin, new_rows-crop_margin))
#enhanced_img_contrast = np.array(enhanced_img_contrast)


if(0):
    print('saving the image')
    scipy.misc.imsave('../enhanced/' + img_name,enhanced_img);
else:
    # Original image
    plt.imshow(img, cmap='Greys_r')
    plt.show()
    # Gabor filtered image
    plt.imshow(enhanced_img,cmap = 'Greys_r');
    plt.show()
    # Contrast increased and Gabor filtered
#    plt.imshow(enhanced_img_contrast,cmap = 'Greys_r');
#    plt.show()
    enhanced_img_contrast.show()
    
    
