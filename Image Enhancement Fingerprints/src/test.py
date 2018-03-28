# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:50:45 2018

@author: khanc
"""

import numpy as np
#import numpy as np;
import matplotlib.pylab as plt;
import scipy.ndimage
import sys
import imageio as imio
import os as os

from image_enhance import image_enhance

#%% 
img_name = '1.jpg'
img = scipy.ndimage.imread('../images/' + img_name);

plt.imshow(img)
plt.show()

e_img1 = image_enhance(img)

plt.imshow(e_img1)
plt.show()

#%%

#im1 = os.path.dirname(os.path.abspath("C:/Users/khanc/Desktop/Image Enhancement Fingerprints/Fingerprint-Enhancement-Python/images/GT_000000001-01-065.png"))
#
#img1 = imio.imread(im1)
#img1 = imio.imread("C:\Users\khanc\Desktop\Image Enhancement Fingerprints\Fingerprint-Enhancement-Python\images\GT_000000001-02-033.png")
#
