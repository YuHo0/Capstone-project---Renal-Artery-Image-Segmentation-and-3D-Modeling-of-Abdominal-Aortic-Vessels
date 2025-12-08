import cv2
import numpy as np
import os

def gamma_correction(image, gamma):
    p_min = np.min(image)
    p_max = np.max(image)
    gamma_corrected_image = np.array(((image - p_min) / (p_max - p_min)) ** gamma * 255, dtype="uint8")
    return gamma_corrected_image

def otsu_thresholding(image):
    pixel_number = image.shape[0] * image.shape[1]
    mean_weights = 1.0 / pixel_number
    his, bins = np.histogram(image, np.arange(0, 257))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]:
        Wb = np.sum(his[:t])
        Wf = np.sum(his[t:])
        if Wb == 0 or Wf == 0:
            continue
        mb = np.sum(his[:t] * np.arange(0, t)) / Wb
        mf = np.sum(his[t:] * np.arange(t, 256)) / Wf
        value = Wb * Wf * (mb - mf) ** 2
        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = image.copy()
    final_img[image > final_thresh] = 255
    final_img[image <= final_thresh] = 0
    return final_img

def largest_contour_extraction(binary_image):
    num_labels, labels_im = cv2.connectedComponents(binary_image)
    label_areas = [np.sum(labels_im == i) for i in range(num_labels)]
    largest_label = np.argmax(label_areas[1:]) + 1
    mask = np.zeros(binary_image.shape, dtype="uint8")
    mask[labels_im == largest_label] = 255
    return mask

def extract_trunk_contour(image_path, output_path, gamma=0.1):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gamma_corrected_image = gamma_correction(image, gamma)
    binary_image = otsu_thresholding(gamma_corrected_image)
    mask = largest_contour_extraction(binary_image)
    trunk_contour_image = cv2.bitwise_and(image, image, mask=mask)
    base_filename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_path, base_filename), trunk_contour_image)

def process_folder(input_folder, output_folder, gamma=0.1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".bmp"):
            image_path = os.path.join(input_folder, filename)
            extract_trunk_contour(image_path, output_folder, gamma)
