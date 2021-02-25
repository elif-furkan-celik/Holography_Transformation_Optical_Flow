#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:03:13 2018

@author: bahadir
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import random


def homography(crr):
    max_in = []
    H = None

    for i in range(1000):
        crr1 = crr[random.randrange(0, len(crr))]
        crr2 = crr[random.randrange(0, len(crr))]
        Four = np.vstack((crr1, crr2))
        crr3 = crr[random.randrange(0, len(crr))]
        Four = np.vstack((Four, crr3))
        crr4 = crr[random.randrange(0, len(crr))]
        Four = np.vstack((Four, crr4))

        aList = []
        h = None
        for corr in Four:
            p1 = np.matrix([corr.item(0), corr.item(1), 1])
            p2 = np.matrix([corr.item(2), corr.item(3), 1])

            a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
            a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
            aList.append(a1)
            aList.append(a2)

        matrixA = np.matrix(aList)
        u, s, v = np.linalg.svd(matrixA)

        h = np.reshape(v[8], (3, 3))

        h = (1/h.item(8)) * h

        in_ = []

        for c in range(len(crr)):

            p1 = np.transpose(np.matrix([crr[c][0].item(0), crr[c][0].item(1), 1]))
            estimatep2 = np.dot(h, p1)
            estimatep2 = (1/estimatep2.item(2))*estimatep2

            p2 = np.transpose(np.matrix([crr[c][0].item(2), crr[c][0].item(3), 1]))
            error = p2 - estimatep2
            d = np.linalg.norm(error)

            if d < 5:
                in_.append(crr[c])

        if len(in_) > len(max_in):
            max_in = in_
            H = h
        print("Iteration: ", i ,"Crr size: ", len(crr), " Number of Inliers: ", len(in_), "Number of Max inliers: ", len(max_in))

    return H, max_in

def drawMatches(img1, kp1, img2, kp2, matches, inliers = None):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out

def warp_p(img, M, dsize):
    H,V,C = img.shape
    mtr = np.zeros((V,H,C), dtype='int')
    for i in range(img.shape[0]):
        mtr[:,i] = img[i]

    R,C = dsize
    dst = np.zeros((R,C,mtr.shape[2]))
    for i in range(mtr.shape[0]):
        for j in range(mtr.shape[1]):
            res = np.squeeze(np.asarray(np.dot(M, [i,j,1])))
            i2,j2,_ = (res * (res[0] / res[2] + 0.5)).astype(int)
            if i2 >= 0 and i2 < R:
                if j2 >= 0 and j2 < C:
                    dst[i2,j2] = mtr[i,j]

    V,H,C = dst.shape
    img2 = np.zeros((H,V,C), dtype='int')
    for i in range(dst.shape[0]):
        img2[:,i] = dst[i]

    return img2

METHODS =['SIFT','SURF', 'ORB']

# Read the images
img1 = cv2.imread('input/img1.ppm')
img2 = cv2.imread('input/img2.ppm')
#cv2.imshow('Input',img1)

# Convert the images to grayscale
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

for METHOD in METHODS:
    if METHOD == 'SIFT':
        print('Calculating SIFT features...')
        
        # Create a SIFT object
        sift = cv2.xfeatures2d.SIFT_create()

        # Get the keypoints and the descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img1_gray,None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2_gray,None)
        # keypoints object includes position, size, angle, etc.
        # descriptors is an array. For sift, each row is a 128-length feature vector

    elif METHOD == 'SURF':
        print('Calculating SURF features...')
        surf = cv2.xfeatures2d.SURF_create(4000)
        keypoints1, descriptors1 = surf.detectAndCompute(img1_gray,None)
        keypoints2, descriptors2 = surf.detectAndCompute(img2_gray,None)

    elif METHOD == 'ORB':
        print('Calculating ORB features...')
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(img1_gray,None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2_gray,None)     
        # Note: Try cv2.NORM_HAMMING for this feature
        
    # Draw the keypoints
    img1 = cv2.drawKeypoints(image=img1, outImage=img1, keypoints=keypoints1, 
                            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                            color = (0, 0, 255))

    img2 = cv2.drawKeypoints(image=img2, outImage=img2, keypoints=keypoints2, 
                            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                            color = (0, 0, 255))

    # Display the images
    #cv2.imshow('Keypoints 1', img1)
    #cv2.imshow('Keypoints 2', img2)

    # Create a brute-force descriptor matcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 
    # Different distances can be used.

    # Match keypoints
    matches1to2 = bf.match(descriptors1,descriptors2)
    # matches1to2 is a DMatch object
    # LOOK AT OPENCV DOCUMENTATION AND 
    #   LEARN ABOUT THE DMatch OBJECT AND ITS FIELDS, SPECIFICALLY THE STRENGTH OF MATCH
    #   matches1to2[0].distance

    # Sort according to distance and display the first 40 matches
    matches1to2 = sorted(matches1to2, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches1to2[:40],img2,flags=2)
    plt.imshow(img3)
    #plt.show()
    cv2.imwrite('old_matches({}).png'.format(METHOD), img3)
   
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1,descriptors2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    
    img4 = cv2.drawMatches(img1,keypoints1,img2,keypoints2, sorted(good, key = lambda x:x.distance)[:55],img2,flags=2)
    plt.imshow(img4)
    #plt.show()
    cv2.imwrite('best_matches_first_50({}).png'.format(METHOD), img4)
   
    img10 = cv2.drawMatches(img1,keypoints1,img2,keypoints2, sorted(good, key = lambda x:x.distance),img2,flags=2)
    plt.imshow(img10)
    #plt.show()
    cv2.imwrite('best_matches_all({}).png'.format(METHOD), img10)
    
    print("Rate of " + METHOD + " is: " , len(good)/len(matches1to2))

    correspondenceList = []
    
    for m in good:
        (x1, y1) = keypoints1[m.queryIdx].pt
        (x2, y2) = keypoints2[m.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])
    
    crr = np.matrix(correspondenceList)
    
    homog, mask = homography(crr)

    img5 = drawMatches(img1_gray, keypoints1, img2_gray, keypoints2, good, mask)
    plt.imshow(img5)
    #plt.show()
    cv2.imwrite('homography_mask({}).png'.format(METHOD), img5)

    img6 = warp_p(img1, homog, (img1.shape[1], img1.shape[0]))
    plt.imshow(img6)
    #plt.show()
    cv2.imwrite('wrapped({}).png'.format(METHOD), img6)

    print("Rate of " + METHOD + " is: " , len(good)/len(matches1to2))