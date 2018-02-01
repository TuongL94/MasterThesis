# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:32:42 2018

@author: Tuong Lam
"""

import numpy as np
import tensorflow as tf

x = np.array([3,4])
y  = np.array([3,2])
#b = tf.pow(x-y,2)
#print(b)
#with tf.Session() as sess:
#    print(b.eval())
c = (x==y)
print(c)
