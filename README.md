# Automated Renal Artery Segmentation and 3D Modeling System 🩺

An integrated software system for automated medical image analysis, featuring deep learning-based segmentation (U-Net), object detection (YOLO), and 3D reconstruction of abdominal aortic vessels.

## 🏆 Project Highlights

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-YOLOv7%20%7C%20U--Net-orange?style=flat-square)
![System Status](https://img.shields.io/badge/Status-Academic_Capstone_Project-lightgrey?style=flat-square)
![Code Link](https://img.shields.io/badge/Code_Base-View_on_GitHub-success?style=flat-square)

## 📖 Executive Summary

**"Bridging the gap between 2D medical imaging and 3D clinical assessment."**

This project presents an integrated software system designed to assist radiologists and surgeons in the diagnosis of **Abdominal Aortic Aneurysm (AAA)**. By processing raw Computed Tomography (CT) slices, the system automates the segmentation of the renal artery and abdominal aorta, ultimately reconstructing high-fidelity **3D STL models**.

The core innovation lies in its hybrid pipeline, combining traditional image processing (for anatomical landmark detection) with state-of-the-art deep learning models (YOLO and U-Net) to achieve precise vascular extraction.

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
* **Purpose:** Noise reduction and automatic localization of the Region of Interest (ROI).
* **Techniques:** **Otsu's Binarization** and **Connected Component Labeling (CCL)** are used for initial noise filtering and anatomical landmark identification (e.g., Spine Extraction).
* **Slicing Filter:** A trained **YOLOv7 Object Detection** model is implemented to detect the **Kidney ROI**, effectively filtering irrelevant CT slices and retaining only those containing the aorta-renal junction.

### Phase 2: Semantic Segmentation (U-Net)
* **Purpose:** Precise, pixel-level extraction of vessel contours.
* **Architecture:** Custom **U-Net** Convolutional Neural Network.
* **Inference:** The model predicts binary masks for the abdominal aorta and renal arteries on ROI-cropped images from Phase 1.

### Phase 3: 3D Reconstruction & Modeling
* **Algorithm:** The system stacks the predicted 2D masks along the Z-axis (longitudinal axis) based on slice thickness. The boundary contours are then used to generate a **triangular mesh surface** (STL format) for solid modeling.

## 📝 Author's Note

The full academic report is available in the `docs/` folder (in Traditional Chinese). This repository, including the code structure and this README, serves as the English technical summary demonstrating my skills in deep learning implementation, computer vision, and full-stack system integration.
