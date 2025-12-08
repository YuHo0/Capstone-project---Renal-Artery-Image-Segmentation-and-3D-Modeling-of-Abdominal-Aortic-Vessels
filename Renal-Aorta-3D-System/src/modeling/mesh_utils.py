import cv2
import numpy as np
import glob
from stl import mesh
from itertools import combinations
import math


#read file name
dirPath = r"C:\Users\USER\Desktop\3d\stl\test\*.png"


vertices = []



for index in range(len(glob.glob(dirPath))):
    img = cv2.imread(glob.glob(dirPath)[index], cv2.IMREAD_GRAYSCALE)
    img_edge = cv2.Canny(img, 30, 100) #論文內是做一次erosion後，原圖減去取得邊緣
   
    vertices.append([])
    for i in range(len(img_edge)): #y
        
        for j in range(len(img_edge[i])): #x
            if(img_edge[i][j] == 255):
                vertices[index].append([j,i,(-1)*index*20])

vertices = np.array(vertices)
print("=============================")
print(len(vertices))

faces = []

for k in range(len(vertices)-1):
    for i in range(len(vertices[k])): #上層對下層
        temp = []
        for j in range(len(vertices[k+1])):
                x = abs(vertices[k+1][j][0]-vertices[k][i][0])
                y = abs(vertices[k+1][j][1]-vertices[k][i][1])
                sum = x*x + y*y + 400
                temp.append([j,sum])

        temp.sort(key=lambda s: s[1])

        test = list(combinations([temp[0][0],temp[1][0],temp[2][0],temp[3][0],temp[4][0]],2))
        print(test[0][0])
        
        for z in range(len(test)):
            faces.append([vertices[k][i],vertices[k+1][test[z][0]],vertices[k+1][test[z][1]]])
    for i in range(len(vertices[k+1])): #上層對下層
        temp = []
        for j in range(len(vertices[k])):
                x = abs(vertices[k][j][0]-vertices[k+1][i][0])
                y = abs(vertices[k][j][1]-vertices[k+1][i][1])
                sum = x*x + y*y + 400
                temp.append([j,sum])

        temp.sort(key=lambda s: s[1])

        test = list(combinations([temp[0][0],temp[1][0],temp[2][0],temp[3][0],temp[4][0]],2))
        print(test[0][0])
        
        for z in range(len(test)):
            faces.append([vertices[k+1][i],vertices[k][test[z][0]],vertices[k][test[z][1]]])
    

faces = np.array(faces)
print(faces.shape)      


m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i in range(faces.shape[0]):
    for j in range(3):
        m.vectors[i][j] = faces[i][j]
m.save('tTt.stl')

