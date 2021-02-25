#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:03:13 2018

@author: bahadir
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2

METHOD = 'SIFT' # 'SIFT','SURF', 'ORB' 

# Read the images
img1 = cv2.imread('input/img1.ppm')
img2 = cv2.imread('input/img2.ppm')
#cv2.imshow('Input',img1)

# Convert the images to grayscale
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(prev=img1_gray,
                                    next=img2_gray, flow=None,
                                    pyr_scale=0.8, levels=15, winsize=5,
                                    iterations=10, poly_n=5, poly_sigma=0,
                                    flags=10)

h, w = flow.shape[:2]
flow = -flow
flow[:,:,0] += np.arange(w)
flow[:,:,1] += np.arange(h)[:,np.newaxis]
img7 = cv2.remap(img1, flow, None, cv2.INTER_LINEAR)

plt.imshow(img7)
plt.show()
cv2.imwrite('calcOpticalFlowFarneback.png', img7)

# params for corner detection 
feature_params = dict( maxCorners = 100, 
                    qualityLevel = 0.3, 
                    minDistance = 7, 
                    blockSize = 7 ) 

# Parameters for lucas kanade optical flow 
lk_params = dict( winSize = (15, 15), 
                maxLevel = 2, 
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                            10, 0.03)) 

p0 = cv2.goodFeaturesToTrack(img1_gray, mask = None, 
                            **feature_params) 



# calculate optical flow 
p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, 
                                        img2_gray, 
                                        p0, None, 
                                        **lk_params) 

# Select good points 
good_new = p1[st == 1] 
good_old = p0[st == 1] 
mask = np.zeros_like(img1_gray) 
color = np.random.randint(0, 255, (100, 3)) 
# draw the tracks 
for i, (new, old) in enumerate(zip(good_new,  
                                    good_old)): 
    a, b = new.ravel() 
    c, d = old.ravel() 
    mask = cv2.line(mask, (a, b), (c, d), 
                    color[i].tolist(), 2) 
        
    frame = cv2.circle(img1_gray, (a, b), 5, 
                        color[i].tolist(), -1) 
        

img8 = cv2.add(frame, mask) 
plt.imshow(img8)
plt.show()
cv2.imwrite('calcOpticalFlowPyrLK.png', img8)

