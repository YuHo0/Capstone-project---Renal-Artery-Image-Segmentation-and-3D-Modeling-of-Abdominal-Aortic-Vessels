# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:16:47 2021

@author: Ubuntu
"""

import cv2
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
import numpy as np

def otsu_implementation(img_title="5.bmp", is_normalized=False, is_reduce_noise=False):
    # 使用灰度模式讀取影象
    image = cv2.imread(img_title, 0)

    # 是否使用高斯模糊
    # 這裡不使用
    if is_reduce_noise:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # 設定直方圖bin的數量
    bins_num = 256

    # 獲取影象的直方圖
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # 是否對影象進行正則化
    # 這裡不進行
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # 計算bin的中心
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # 迭代累加得到權值
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # 計算得到均值
    mean1 = np.cumsum(hist * bin_mids) / weight1
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    # 計算最終方差
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # 取最大值對應的下標
    index_of_max_val = np.argmax(inter_class_variance)
    # 從bin中取得閾值
    threshold = bin_mids[:-1][index_of_max_val]
    print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

def call_otsu_threshold(img_title="5.bmp", is_reduce_noise=False):
    # 讀取二值影象
    image = cv2.imread(img_title, 0)

    if is_reduce_noise:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # 現實影象的直方圖
    plt.hist(image.ravel(), 256)
    plt.xlabel('Colour intensity')
    plt.ylabel('Number of pixels')
    #plt.savefig("image_hist.png")
    plt.show()
    plt.close()

    # 影象二值化，其中THRESH_OTSU就為大津演算法
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    print("Obtained threshold: ", otsu_threshold)

    # 顯示兩類大概率的直方圖
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(image_result.ravel(), 256)
    ax.set_xlabel('Colour intensity')
    ax.set_ylabel('Number of pixels')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: ('%1.1fM') % (x*1e-6)))
    #plt.savefig("image_hist_result.png")
    plt.show()
    plt.close()

    # 將二值化後的影象顯示
    cv2.imshow("Otsu's thresholding result", image_result)
    cv2.imwrite("res.bmp",image_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    call_otsu_threshold()
    otsu_implementation()


if __name__ == "__main__":
    main()
