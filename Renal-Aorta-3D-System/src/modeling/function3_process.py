import cv2
import numpy as np
import glob
from stl import mesh
from itertools import combinations
import re
import os

def generate_stl(dirPath, output_dir):
    vertices = []
    files = glob.glob(dirPath)
    for index, file_path in enumerate(files):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # 如果文件無法讀取，跳過此次循環
        img_edge = cv2.Canny(img, 30, 100)
        vertex_layer = []
        for i in range(len(img_edge)):
            for j in range(len(img_edge[i])):
                if img_edge[i][j] == 255:
                    vertex_layer.append([j, i, (-1) * index * 20])
        vertices.append(vertex_layer)

    vertices = np.array(vertices, dtype=object)
    faces = []

    for k in range(len(vertices) - 1):
        # Process upper to lower layer
        for i in range(len(vertices[k])):
            temp = []
            for j in range(len(vertices[k+1])):
                x = abs(vertices[k+1][j][0] - vertices[k][i][0])
                y = abs(vertices[k+1][j][1] - vertices[k][i][1])
                sum = x*x + y*y + 400
                temp.append([j, sum])

            temp.sort(key=lambda s: s[1])
            test = list(combinations([temp[0][0], temp[1][0], temp[2][0], temp[3][0], temp[4][0]], 2))
            
            for z in range(len(test)):
                faces.append([vertices[k][i], vertices[k+1][test[z][0]], vertices[k+1][test[z][1]]])
        
        # Process lower to upper layer
        for i in range(len(vertices[k+1])):
            temp = []
            for j in range(len(vertices[k])):
                x = abs(vertices[k][j][0] - vertices[k+1][i][0])
                y = abs(vertices[k][j][1] - vertices[k+1][i][1])
                sum = x*x + y*y + 400
                temp.append([j, sum])

            temp.sort(key=lambda s: s[1])
            test = list(combinations([temp[0][0], temp[1][0], temp[2][0], temp[3][0], temp[4][0]], 2))
            
            for z in range(len(test)):
                faces.append([vertices[k+1][i], vertices[k][test[z][0]], vertices[k][test[z][1]]])

    faces = np.array(faces)
    print(f"面數量和形狀: {faces.shape}")  # 列印面數量和形狀

    # 創建 Mesh 對象並保存 STL 文件
    m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i in range(faces.shape[0]):
        m.vectors[i] = np.array(faces[i])
    
    # 保存 STL 文件至目錄並防止檔名重覆
    stl_path = save_stl_with_incremental_names(output_dir, 'patient', '.stl')
    m.save(stl_path)
    print(f"STL 文件已生成於 {stl_path}")

def save_stl_with_incremental_names(directory, filename="patient", ext=".stl"):
    # 構建正則表達式匹配文件名
    pattern = re.compile(rf"{filename}_(\d+){ext}$")
    max_num = 0
    for f in os.listdir(directory):
        match = pattern.match(f)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num

    # 生成新的文件名
    new_filename = f"{filename}_{max_num + 1}{ext}"
    return os.path.join(directory, new_filename)

def processImages(dirPath):
    output_dir = 'D:\\GUI_code\\data\\3d'  # 定義輸出目錄
    generate_stl(dirPath, output_dir)  # 直接調用函數，傳遞兩個參數

# 調用 processImages 函數，確保路徑正確
processImages('C:\\Users\\USER\\Desktop\\3d\\stl\\test\\*.png')
