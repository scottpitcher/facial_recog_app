# TensorFlow Face Recognition App

This project is a robust face‐matching system that leverages a **Siamese CNN** and **OpenCV/MediaPipe** preprocessing to verify identities by comparing face embeddings. It isolates the face region to eliminate background bias and achieves high accuracy even in previously unseen environments.

Check out the full report and demo [here!](https://yourrepo.github.io/face_recognition_demo)

## 🔍 Overview
- **Captures** input images or live video streams
  - Detects faces with **MediaPipe FaceMesh** or **OpenCV Haar cascades** 
- **Crops** to the face-only region and **masks out** all background via segmentation 
- **Embeds** each face crop using a pretrained **FaceNet**‐style CNN 
- **Compares** embeddings using cosine‐distance thresholding in a Siamese architecture 
- **Experiments** include:
  - Original vs. face-only anchor evaluation
  - Background augmentation during training to enforce invariance
  - Grad-CAM visualizations to inspect model focus *(notebooks/gradcam_analysis.ipynb)*
- **Automated** evaluation on custom test sets for continuous benchmarking *(scripts/evaluate.py)*

---
## 💻 Example Usage
<sub><i>*Note: screenshots generated on custom test images.*</i></sub>

Example 1: Matching with original anchor  
![Original Anchor Match](images/original_match.png)  
Example 2: Matching with face-only anchor  
![Face-Only Anchor Match](images/face_only_match.png)

---
## 🔨 Roadblocks + Solutions
| Roadblock                                    | Solution                                                                                                |
|----------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Inconsistent embedding focus                 | Use Grad-CAM to verify and retrain network to concentrate on central facial landmarks                    |
| High false negatives when background changes | Crop and mask out the background; train with random background augmentation                              |

<sub><i>*All “Potential additions” are plan items for future work.*</i></sub>

## Before and After Facial Cropping

## ⚙️ Features

- 👀 **Face Detection & Cropping** (OpenCV)  
- 🎭 **Background Masking** via segmentation  
- 🤝 **Siamese Embedding Comparison** with cosine similarity  
- 📈 **Threshold‐Based Identity Verification**  
- 🔍 **Grad-CAM Interpretability** tools  
- ⚙️ **Automated Benchmarks** on new environments  

---
## 🛠️ Tech Stack

| Component        | Tool                                             |
|------------------|--------------------------------------------------|
| Language         | Python                                           |
| Face Detection   | MediaPipe FaceMesh, OpenCV Haar Cascades         |
| Segmentation     | U²-Net / BiSeNet                                 |
| Embeddings       | TensorFlow, FaceNet‐style CNN                    |
| Visualization    | Matplotlib, Grad-CAM                             |
| Data Storage     | JSON                                             |

---

### 📥 Example Usage

```bash
python main.py --mode match \
  --anchor images/anchor_face.png \
  --input images/query_face.jpg \
  --threshold 0.6
