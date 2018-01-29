# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:32:42 2018

@author: Tuong Lam
"""

import numpy as np
import tensorflow as tf

x = 4
y1  = tf.constant(x,4,name = 'h1')

with tf.Session() as sess:
    sess.run(y1)
    print(y1.eval())