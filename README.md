# Automated Renal Artery Segmentation and 3D Modeling System 🩺

An integrated software system for automated medical image analysis, featuring deep learning-based segmentation (U-Net), object detection (YOLOv4/v7), and 3D reconstruction of abdominal aortic vessels.

## 🏆 Project Highlights

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-YOLOv4%20%26%20v7%20%7C%20U--Net-orange?style=flat-square)
![System Status](https://img.shields.io/badge/Status-Academic_Capstone_Project-lightgrey?style=flat-square)
![Code Link](https://img.shields.io/badge/Code_Base-View_on_GitHub-success?style=flat-square)

## 📖 Executive Summary

**"Bridging the gap between 2D medical imaging and 3D clinical assessment."**

This project presents an integrated software system designed to assist radiologists and surgeons in the diagnosis of **Abdominal Aortic Aneurysm (AAA)**. [cite_start]By processing raw Computed Tomography (CT) slices, the system automates the segmentation of the renal artery and abdominal aorta, ultimately reconstructing high-fidelity **3D STL models**[cite: 6].

[cite_start]The core innovation lies in its hybrid pipeline, combining traditional image processing (for anatomical landmark detection) with state-of-the-art deep learning models (**YOLOv4/v7** and **U-Net**) to achieve precise vascular extraction[cite: 4, 5].

## 📸 Visual Demonstration

### 1. System GUI Interface
**(This screenshot demonstrates software engineering and user interface skills)**

<p align="center">
  <img src="/Renal-Aorta-3D-System/assets/GUI_SCREENSHOT_LINK1.png" alt="Graphical User Interface Screenshot" width="300"/>
  <img src="/Renal-Aorta-3D-System/assets/GUI_SCREENSHOT_LINK2.png" alt="Graphical User Interface Screenshot" width="300"/>
  <img src="/Renal-Aorta-3D-System/assets/GUI_SCREENSHOT_LINK3.png" alt="Graphical User Interface Screenshot" width="300"/>
  <br>
  <em>Figure 1: The main graphical user interface (GUI) developed with PyQt, allowing users to load DICOM series and execute the analysis pipeline.</em>
</p>

### 2. Final 3D Reconstruction Result
**(This screenshot demonstrates the final clinical outcome and complex modeling)**

<p align="center">
  <img src="/Renal-Aorta-3D-System/assets/THREE_D_MODEL_LINK1.png" alt="Final 3D Aorta Model" width="400"/>
  <img src="/Renal-Aorta-3D-System/assets/THREE_D_MODEL_LINK2.jpg" alt="Final 3D Aorta Model" width="400"/>
  <br>
  <em>Figure 2: Final 3D reconstruction of the abdominal aorta and renal arteries from segmented slices.</em>
</p>


## ⚙️ Technical Architecture & Methodology

Since the full academic report (`docs/`) is in Traditional Chinese, this section provides a comprehensive English summary of the implemented algorithms for technical reviewers.

### Phase 1: Intelligent Preprocessing & ROI Extraction
* [cite_start]**Data Transformation:** Extracted CT slices from **DICOM format** and performed **linear transformations** to generate high-quality BMP format images.
* [cite_start]**Enhancement:** Applied contrast adjustment and **Otsu's Binarization** to achieve a clear presentation of abdominal vascular structures[cite: 3].
* [cite_start]**Slicing Filter (Kidney Localization):** Employed both **YOLOv4 and YOLOv7** object detection models to identify and select specific slices containing the aorta and renal arteries, effectively filtering out irrelevant data.

### Phase 2: Semantic Segmentation (U-Net)
* **Purpose:** Precise, pixel-level extraction of vessel contours.
* [cite_start]**Architecture:** Custom **U-Net** Convolutional Neural Network[cite: 5].
* [cite_start]**Inference:** The model predicts binary masks for the **abdominal aorta** and the critical **connection between the aorta and renal arteries** on ROI-cropped images[cite: 5].

### Phase 3: 3D Reconstruction & Modeling
* [cite_start]**Algorithm:** The system stacks the predicted 2D masks along the Z-axis (longitudinal axis) based on slice thickness to create voxel points[cite: 6].
* [cite_start]**Mesh Generation:** Algorithms convert the voxel data into a **triangular mesh surface** to construct a standard **STL format 3D model**, ready for preoperative evaluation[cite: 6].

## 🛠️ Key Libraries & Tools
The system was built using the following core technologies:
* **Interface:** PyQt5
* **Deep Learning:** PyTorch (U-Net), Darknet (**YOLOv4 & YOLOv7**)
* **Image Processing:** OpenCV, NumPy, SimpleITK
* **3D Modeling:** NumPy-STL

## 📝 Author's Note

The full academic report is available in the `docs/` folder (in Traditional Chinese). This repository, including the code structure and this README, serves as the English technical summary demonstrating my skills in deep learning implementation, computer vision, and full-stack system integration.
