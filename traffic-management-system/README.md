# Hardware-Based Intelligent Traffic Management System

[![PyTorch](https://img.shields.io/badge/PyTorch-latest-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-enabled-green.svg)]()
[![Edge AI](https://img.shields.io/badge/edge-Jetson%20Nano-76B900.svg)]()

## Overview

A real-time computer vision system for **adaptive traffic signal control** using CNN-based vehicle detection deployed on edge hardware (NVIDIA Jetson Nano). The system continuously analyses camera feeds, estimates per-lane congestion, and adjusts signal timing to reduce average wait time.

---

## Research Questions

1. How can vision systems maintain high accuracy across varying environmental conditions (lighting, weather, seasonality)?
2. What is the role of **model interpretability** in safety-critical visual systems?
3. Can we optimise CNN inference speed sufficiently for **embedded hardware deployment** without compromising accuracy?

---

## Methods

| Component | Choice |
|-----------|--------|
| Backbone architecture | Custom CNN built on **ResNet-50** |
| Framework | **PyTorch** |
| Edge hardware | **NVIDIA Jetson Nano** |
| Frame processing | **OpenCV** for real-time capture and pre-processing |
| Training strategy | Transfer learning + targeted data augmentation |

---

## Results

| Metric | Value |
|--------|-------|
| Classification accuracy (unseen test set) | **92%** |
| Per-frame inference latency | **< 50 ms** |
| Deployment target | Successfully deployed on embedded hardware |
| Real-world impact | **~15% reduction** in average vehicle wait time |

---

## Key Insights

- **Challenges encountered:** Model degradation in low-light conditions and across seasonal variation in scene appearance.
- **Solutions implemented:** Aggressive data augmentation (lighting, blur, weather), transfer learning from a pre-trained ResNet-50 backbone, and ensemble methods at the decision layer.
- **Future research direction:** Explainability analysis using **saliency maps** and attention visualisation to understand which image regions drive the model's decisions — a critical requirement for safety-critical deployment.

---

## Related Work

- YOLO family for real-time object detection
- Inference optimisation on edge devices (TensorRT, quantisation)
- Explainable AI in autonomous and safety-critical systems

---

## Technologies

`PyTorch` · `OpenCV` · `CUDA` · `NVIDIA Jetson Nano` · `CNN` · `Transfer Learning` · `ResNet-50`

---

## Future Directions

- Saliency- and attention-based interpretability layer
- Domain adaptation for cross-city / cross-climate deployment
- INT8 quantisation to further reduce inference latency
- Comparison with YOLO-based one-stage detectors

---

## Repository Structure

```
traffic-management-system/
├── README.md
├── requirements.txt
├── notebooks/      # experiments, ablations, error analysis
├── src/            # training and inference code
├── results/        # metrics, plots, confusion matrices
└── data/           # (or link to data source)
```

---

## References & Citations

- Adaptive traffic control via computer vision (literature review pending)
- ResNet: Deep Residual Learning for Image Recognition (He et al., 2016)
- Explainable AI for vision: Grad-CAM, saliency methods
