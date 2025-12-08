# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import glob

#read file name
dirPath = r"C:\Users\Ubuntu\Desktop\todo\*.bmp"


#取r參數
for index in range(len(glob.glob(dirPath))):
    #read img
    
    
    img = cv2.imread(glob.glob(dirPath)[index], cv2.IMREAD_GRAYSCALE);
    img_data = np.asarray(img,dtype = 'float16');
    img_data_r = np.array(img,dtype = 'float16');
    Pmin = img_data_r[0][0];
    Pmax = img_data_r[0][0];
    for i in range(len(img_data_r)):
      for j in range(len(img_data_r[0])):
        if(img_data_r[i][j] < Pmin):
            Pmin = img_data_r[i][j];
        if(img_data_r[i][j] > Pmax):
            Pmax = img_data_r[i][j];    
    print("最高最低:", Pmin,Pmax);
    for i in range(len(img_data_r)):
      for j in range(len(img_data_r[0])):
          #img_data_r[i][j] = (((img_data_r[i][j] - Pmin)/(Pmax - Pmin))**1.5)*255
          a = img_data_r[i][j] - Pmin
          b = Pmax - Pmin
          img_data_r[i][j] = ((a/b)**0.1)*255
    img_data_i = img_data_r.astype('uint8')
    #cv2.imshow("Mask", img_data_i)
   # cv2.imwrite('%s.bmp'%(index), img_data_i);
     
    
    #canny = cv2.Canny(cv2.GaussianBlur(img_data_i, (5, 5), 0), 30, 150);
    #cv2.imshow('cannyImg',canny);
    #cv2.imwrite('%s.bmp'%(index), canny);
    
    binary, contours, hierarchy = cv2.findContours(img_data_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    
    print(max_idx)
    mask = np.zeros(img.shape, dtype="uint8")  #依Contours圖形建立mask

    cv2.drawContours(mask, contours, max_idx, 255, -1) #255        →白色, -1→塗滿
    result = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow("Mask", mask)
    #cv2.imwrite('%s.bmp'%(index), mask);
    #將mask與原圖形作AND運算
    #cv2.imshow("Image + Mask", result)
    
    cv2.imwrite('./todo/%s.bmp'%(index), result)
    cv2.imwrite('./todo/%s.bmp'%(index), result)
    #cv2.imwrite('%s.bmp'%(index), img_data_i)

    cv2.waitKey(0)     
    
    #cv2.imwrite('%s.bmp'%(index), img_data);
   

    
