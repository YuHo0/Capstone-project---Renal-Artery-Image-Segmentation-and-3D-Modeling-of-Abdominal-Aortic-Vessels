# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:30:04 2021

@author: Ubuntu
"""

from __future__ import print_function
from skimage import measure
import numpy as np
import cv2

image = cv2.imread("res.bmp", 0)
img = cv2.imread("IMG-0001-00001.bmp",0)
bodyimg = cv2.imread("./todo/result.bmp",0)
   
ret, labels_1st, stats, centroid = cv2.connectedComponentsWithStats(image,connectivity=8)
print(stats)
area = [0]
print(area) 
for (i, label) in enumerate(np.unique(labels_1st)):
    if label == 0:
        continue
    labelMask = np.zeros(image.shape, dtype="uint8")
    labelMask[labels_1st == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    area.append(numPixels)
    max_idx = np.argmax(np.array(area))
print(area)    
print(max_idx)
mask = np.zeros(image.shape, dtype="uint8")
for (i, label) in enumerate(np.unique(labels_1st)):
    labelMask = np.zeros(image.shape, dtype="uint8")
    labelMask[labels_1st == label] = 255
    if label == max_idx:
        print(max_idx)
        mask = cv2.add(mask, labelMask)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations = 3)


M = cv2.moments(closing)
cX = int(M["m10"]/M["m00"])
cY = int(M["m01"]/M["m00"])
cv2.circle(bodyimg, (cX,cY), 5, 0, -1)
cv2.line(bodyimg, (cX,0), (cX,bodyimg.size), 255, 3)
cv2.line(bodyimg, (0,cY), (bodyimg.size,cY), 255, 3)

cv2.imwrite('test1.bmp',mask)
cv2.imwrite('test.bmp',closing)
cv2.imwrite('cir.bmp' , bodyimg)
cv2.waitKey(0)



"""
imgInfo = mask.shape
height = imgInfo[0]
width = imgInfo[1] 

res = mask.copy()        
for i in range(height):    
    for j in range(width):
        res[i,j] = 255 - res[i,j]
        
cv2.imshow('test.bmp',res)
cv2.waitKey(0)

ret, labels_2nd, stats, centroid = cv2.connectedComponentsWithStats(res)
print(stats)
for (i, label) in enumerate(np.unique(labels_2nd)):
    labelMask = np.zeros(image.shape, dtype="uint8")
    labelMask[labels_2nd == label] = 255
cv2.imwrite('test.bmp',labels_2nd)
cv2.waitKey(0)    
"""