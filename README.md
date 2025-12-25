# Monocular Depth Estimation using a Coarse-to-Fine Deep Network

This repository implements a **monocular depth estimation pipeline** inspired by  
**Eigen et al., _Depth Map Prediction from a Single Image using a Multi-Scale Deep Network_ (CVPR 2014)**.

The project reproduces the **core coarse-to-fine learning paradigm** using a **modern fully convolutional architecture**, trained and evaluated on the **KITTI dataset** with sparse LiDAR depth supervision.
https://arxiv.org/pdf/1406.2283


![Qualitative Results](result.png)

---

## Motivation

Depth estimation from a single RGB image is a fundamental problem in:

- Autonomous Driving (AV perception)
- Robotics & Navigation


Eigen et al. addressed challenges in Monocular depth estimation using a **multi-scale architecture** that separates:
- global depth reasoning, and
- local depth refinement.

This project implements and evaluates that idea in a modern deep learning setting.

---

## Method Overview

### Coarse-to-Fine Architecture

The system consists of **two neural networks trained sequentially**:

#### 1. Coarse Network (Global Geometry)
- Predicts a smooth, low-frequency depth map
- Captures global scene layout (road slope, horizon, relative distances)
- Ignores fine object boundaries by design

#### 2. Fine Network (Local Refinement)
- Takes **RGB image + coarse depth** as input
- Refines object boundaries and local structures
- Preserves global geometry while improving sharpness


---

## Architecture Details

### Coarse Network
- Fully convolutional encoder–decoder
- Strided convolutions to increase receptive field
- Bottleneck encodes global scene geometry
- Bilinear upsampling for dense prediction
- Output: `(1 × H × W)` depth map

### Fine Network
- Shallow convolutional network
- Input channels: RGB (3) + coarse depth (1)
- No spatial downsampling
- Learns local refinements only
- Output: refined depth map at full resolution

> **Note:**  
> We use a **modern fully convolutional reinterpretation** of Eigen’s architecture, replacing large fully connected layers with strided convolutions for improved stability and efficiency.

---

## Loss Function

The model is trained using the **scale-invariant loss** proposed by Eigen et al.:

- Operates in **log-depth space**
- Penalizes relative depth errors
- Removes global scale bias
- Ignores invalid pixels using a depth mask

This loss is critical due to the **scale ambiguity inherent in monocular depth estimation**.

---

## Dataset

### KITTI Dataset
- RGB images from camera `image_02`
- Sparse depth maps obtained via LiDAR projection
- Multiple driving sequences
- Only frames with valid RGB–depth pairs are used

### Preprocessing
- Depth converted from `uint16` to meters (`/256`)
- Invalid depth pixels masked
- Images resized to `224 × 224`
- Depth resized using nearest-neighbor interpolation

---


---

## Training Procedure

1. **Train Coarse Network**
   - Input: RGB image
   - Output: coarse depth map
   - Loss: scale-invariant loss

2. **Freeze Coarse Network**
   - Prevents global geometry drift

3. **Train Fine Network**
   - Input: RGB + coarse depth
   - Output: refined depth
   - Loss: scale-invariant loss

---

## Quantitative Evaluation

### Evaluation Protocol
- Metrics follow Eigen et al.
- Only valid depth pixels are evaluated
- Predictions are **median-scaled per image** to resolve scale ambiguity

### Metrics Reported
- Absolute Relative Error (AbsRel)
- Squared Relative Error (SqRel)
- RMSE
- RMSE (log)
- Accuracy thresholds (δ < 1.25, 1.25², 1.25³)

---

### Coarse + Fine Network Results (KITTI)

| Metric | Value |
|------|------|
| AbsRel | 0.4804 |
| SqRel | 5.8492 |
| RMSE | 13.2679 |
| RMSE (log) | 0.6462 |
| δ < 1.25 | 0.2718 |
| δ < 1.25² | 0.5316 |
| δ < 1.25³ | 0.7358 |

---

## Results Analysis

- The model reliably captures **global scene geometry** such as road slope and horizon depth.
- Fine-scale refinement improves **local depth consistency and edge sharpness**.
- Absolute depth accuracy remains limited due to:
  - Sparse LiDAR supervision
  - Limited dataset size
  - No data augmentation or smoothness regularization

Despite modest quantitative performance, **qualitative improvements from coarse to fine predictions are clearly visible**, validating the effectiveness of the coarse-to-fine design.

---

## Limitations

- Sparse depth supervision
- No data augmentation
- No edge-aware smoothness loss
- Shallow fine network
- No temporal consistency

---

## References

- D. Eigen, C. Puhrsch, R. Fergus  
  *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network*, CVPR 2014

- A. Laina et al.  
  *Deeper Depth Prediction with Fully Convolutional Residual Networks*, 3DV 2016

- C. Godard et al.  
  *Unsupervised Monocular Depth Estimation with Left-Right Consistency*, CVPR 2017

---


