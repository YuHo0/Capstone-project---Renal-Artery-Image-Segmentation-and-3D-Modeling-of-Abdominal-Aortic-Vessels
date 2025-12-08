# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:04:30 2021

@author: Ubuntu
"""
from __future__ import print_function
from skimage import measure
import numpy as np
import cv2

image = cv2.imread("res.bmp", 0)
img = cv2.imread("IMG-0001-00001.bmp",0)

#針對thresholded圖片進行connected components analysis，
# connectivity = 2 表示採用8向方式, background=0表示pixel值為0則認定為背景

labels = measure.label(image, connectivity = 1 , background=0)

#建立一個空的圖，存放稍後將篩選出的字母及數字

mask = np.zeros(image.shape, dtype="uint8")
#顯示一共貼了幾個Lables（即幾個components）

#print("[INFO] Total {} blobs".format(len(np.unique(labels))))

#依序處理每個labels


for (i, label) in enumerate(np.unique(labels)):

        #如果label=0，表示它為背景

    if label == 0:
        print("[INFO] label: 0 (background)")
        continue
    
            #否則為前景，顯示其label編號l
    
    #print("[INFO] label: {} (foreground)".format(i))
    
            #建立該前景的Binary圖
    
    labelMask = np.zeros(image.shape, dtype="uint8")
    
    labelMask[labels == label] = 255
    
            #有幾個非0的像素?
    numPixels = cv2.countNonZero(labelMask)
    area = []
    area.append(0)
    area.append(numPixels)
    
    max_idx = np.argmax(np.array(area))
    
    if label == max_idx:   
        
        mask = cv2.add(mask, labelMask)
      
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1] 

res = mask.copy()        
for i in range(height):    
    for j in range(width):
        res[i,j] = 255 - res[i,j]
mask_r = np.zeros(image.shape, dtype="uint8")     
ret, labels_b, stats, centroid = cv2.connectedComponentsWithStats(res)
print(stats)
for (i, label) in enumerate(np.unique(labels_b)):
    if label == 0:
        continue
    labelMask = np.zeros(image.shape, dtype="uint8")
    labelMask[labels_b == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    mask_r = cv2.add(mask_r, labelMask)
    area = []
    area.append(0)
    area.append(numPixels)
    max_idx = np.argmax(np.array(area))
    if label == max_idx:
        ful = res.copy()
        ful = cv2.add(mask_r, labelMask)
    
for i in range(height):    
    for j in range(width):
        ful[i,j] = 255 - ful[i,j]
result = cv2.bitwise_and(img, ful)
cv2.imshow('test.bmp',result)
cv2.waitKey(0)


 
