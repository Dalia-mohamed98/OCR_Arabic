# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 00:32:51 2019

@author: Mary
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# %matplotlib inline


def init(Dir):
    img  = cv2.imread(Dir)
    dim = (28, 28)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',resized.shape)

    # cv2.imshow('resized',img)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray,127,1,cv2.THRESH_BINARY)

#     plt.imshow(thresh, cmap="gray")
    return (thresh)

#######################################################

def perimeter(thresh):
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    perimeter = cv2.arcLength(cnt,True)
    return round(perimeter,2)

#######################################################

def zoning(gray):
    gridSize = round(28/4)
    zoning = np.zeros(26)
    #divide img to 4x4
    k=0

    for x in range(0,28,gridSize):
        for y in range(0,28,gridSize):
            #process on every zone, get 16 vector
            for m in range(x,gridSize+x):
                for n in range(y,gridSize+y):

                    if(gray[m,n]==0):
                        zoning[k] += 1
            k=k+1      
    # print(zoning)

#sum each row get 4 vector
    for x in range(0,16,4):
        for y in range(x,4+x):

             zoning[k] += zoning[y]
        k+=1
    # print(zoning)

#sum each col get 4 vector
    for x in range(4):
        for y in range(x,16,4):

             zoning[k] += zoning[y]
        k+=1
    # print(zoning)

#sum diagonal get 1 vector
    for x in range(0,16,5):
        zoning[k] += zoning[x]
    # print(zoning)
    k+=1
    #sum anti-diagonal get 1 vector
    for x in range(3,14,3):
    #     print(x)
        zoning[k] += zoning[x]
        
    return zoning

#######################################################


def get_transition(img):
    img_shape = img.shape
    row_transition = []
    col_transition = []
    for row in range(img_shape[0]):
        i = 0
        for col in range(img_shape[1] - 1):
            if(img[row][col] != img[row][col+1]):
                i = i + 1
        col_transition.append(i)
        
    for col in range(img_shape[1]):
        i = 0
        for row in range(img_shape[0] - 1):
            if(img[row][col] != img[row + 1][col]):
                i = i + 1
        row_transition.append(i)
    return max(row_transition),max(col_transition)  

#######################################################

def get_area(img):
    contours,hierarchy = cv2.findContours(img, 1, 2)
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    return area
    
#######################################################
            
def get_features(Dir,closed_loop,connected_comp):
    img  = cv2.imread(Dir,0)
    dim = (28, 28)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    ret , thresh_img = cv2.threshold(resized,np.average(img),255,cv2.THRESH_BINARY)

    feature_vec = []

    # img2 = init(Dir)
    zoning_vec = zoning(thresh_img)

    #add zoning feature
    for zone in zoning_vec:
        feature_vec.append(zone)

    trans_row,trans_col = get_transition(thresh_img)

    #add trnasition feature
    feature_vec.append(trans_row)
    feature_vec.append(trans_col)

    area = get_area(thresh_img)
    perim = perimeter(thresh_img)
    if(perim>0):
        areaToPerRatio = area/perim
    else:
        areaToPerRatio = 0

    #add areaToPerRatio feature
    feature_vec.append(areaToPerRatio)
    #add closed_loop feature
    feature_vec.append(closed_loop)
    #add connected_component feature
    feature_vec.append(connected_comp)

    return feature_vec
