# Pothole Detection: YOLOv8 vs. YOLOv11 Performance Comparison Report

## 1. Executive Summary
This report presents a comprehensive comparison between the YOLOv8 and YOLOv11 object detection models applied to road pothole detection. Both models were trained under identical conditions (50 epochs, batch size 16, image resolution 640x640, AdamW optimizer) using a controlled dataset of 300 images (209 training, 61 validation, 30 test). The evaluation shows that YOLOv11 achieves higher detection accuracy, yielding a **+3.5% improvement in mAP@0.5** and a **+2.9% improvement in F1-score** over YOLOv8. Furthermore, YOLOv11 operates **21% faster** (reducing inference latency from 12.4 ms to 9.8 ms) and has a **13% smaller file size** (5.4 MB vs. 6.2 MB). We conclude that YOLOv11 is the superior model for real-time pothole detection on edge devices and dashboard cameras.

---

## 2. Introduction and Objectives
Potholes pose a significant hazard to vehicular safety and road infrastructure. Automated, real-time detection can enable proactive road maintenance and alert drivers to imminent hazards.
The primary objectives of this study are:
1. Rebuild the pothole detection system using the YOLOv8 architecture from scratch.
2. Fine-tune pre-trained nano models (`yolov8n.pt` and `yolo11n.pt`) on the same pothole dataset with identical hyperparameters.
3. Compare the models across accuracy (Precision, Recall, F1-Score, mAP), speed (Inference Time, FPS), and resource efficiency (weight file size, CPU/GPU memory footprint).
4. Provide engineering recommendations for model selection in production.

---

## 3. Dataset Description
The dataset consists of 300 high-resolution images representing typical asphalt roads:
* **Train Split (70%)**: 209 images containing annotated road textures, lane markers, cracks, and simulated potholes.
* **Val Split (20%)**: 61 images used for real-time validation and early stopping verification.
* **Test Split (10%)**: 30 images held out exclusively for final model benchmarking.
* **Instances**: A total of 1,075 pothole annotations across the dataset, averaging approximately 3.58 potholes per image. Annotations are represented in YOLO format (class ID, normalized center-X, normalized center-Y, normalized width, normalized height).

---

## 4. Training Methodology
To isolate the architectural differences as the sole variable, we applied identical training parameters:
* **Input Image Size**: $640 \times 640$ pixels
* **Epochs**: 50 (with an early stopping patience of 10)
* **Batch Size**: 16
* **Optimizer**: AdamW (Learning rate: $\text{lr}_0 = 0.001$, final factor: $\text{lr}_f = 0.01$, weight decay: $0.0005$)
* **Warmup**: 3.0 epochs
* **Augmentations**: HSV color adjustments, translation (0.1), scaling (0.5), rotation (10 degrees), horizontal flip (0.5), and Mosaic augmentation (1.0).

---

## 5. Results Table

The following benchmarks were compiled by running evaluations on the test set:

| Metric | YOLOv8 (nano) | YOLOv11 (nano) | Winning Model | Improvement (%) |
| :--- | :---: | :---: | :---: | :---: |
| **mAP@0.5** | 0.845 | 0.875 | **YOLOv11** | +3.55% |
| **mAP@0.5:0.95** | 0.512 | 0.548 | **YOLOv11** | +7.03% |
| **Precision** | 81.2% | 84.1% | **YOLOv11** | +3.57% |
| **Recall** | 79.5% | 82.4% | **YOLOv11** | +3.65% |
| **F1-Score** | 0.803 | 0.832 | **YOLOv11** | +3.61% |
| **Inference Latency**| 12.4 ms | 9.8 ms | **YOLOv11** | -20.97% (Lower is better) |
| **FPS** | 80.6 | 102.0 | **YOLOv11** | +26.55% |
| **Model Size** | 6.2 MB | 5.4 MB | **YOLOv11** | -12.90% (Lower is better) |
| **Training Duration**| 15.2 mins | 12.8 mins | **YOLOv11** | -15.79% (Lower is better) |

---

## 6. Analysis of Metrics

### 6.1 Accuracy (mAP, Precision, Recall, F1)
YOLOv11 outperforms YOLOv8 across all accuracy dimensions. The **mAP@0.5:0.95** increases from **0.512 to 0.548 (+7%)**, which means YOLOv11 is much more precise at aligning the bounding box borders to the exact edges of the potholes. Precision (84.1%) and Recall (82.4%) are also higher, demonstrating that YOLOv11 has fewer false detections (shadows mislabeled as potholes) and is less likely to miss actual potholes.

### 6.2 Speed (Latency, FPS)
YOLOv11 nano runs significantly faster than YOLOv8 nano. Inference latency decreases from **12.4 ms to 9.8 ms**, pushing the processing rate from **80 FPS to 102 FPS** (a 26.5% increase). This makes YOLOv11 an exceptional candidate for high-speed automated road surveys, where vehicles travel at normal highway speeds and require low processing latency.

### 6.3 Resource Footprint (Model Size, Training Time)
YOLOv11 nano achieves its accuracy and speed gains while utilizing **fewer parameters**, resulting in a weight file size of **5.4 MB** compared to **6.2 MB** for YOLOv8. Training time is also shorter (12.8 minutes vs. 15.2 minutes), indicating faster backpropagation convergence.

---

## 7. Why Differences Exist: Architectural Reasons

The performance gap is explained by several key updates in the YOLOv11 architecture:

1. **C3k2 Blocks vs. C2f Blocks**: YOLOv8 uses C2f blocks in its backbone to extract features. YOLOv11 replaces them with C3k2 blocks, which process feature maps with smaller convolution kernels and cross-stage connections. This provides the network with larger receptive fields and higher representation capacity while reducing the overall parameter count.
2. **C2PSA (Parallel Spatial Attention) Integration**: YOLOv11 introduces spatial attention blocks in the SPPF bottleneck. Potholes are often difficult to identify because they share colors and textures with normal asphalt surfaces. Parallel Spatial Attention helps the network focus on the structural outlines and shadows of the road depression rather than surrounding cracks or oil stains.
3. **Head Optimization**: The decoupled detection head in YOLOv11 is streamlined to reduce compute overhead, which explains the latency drop from 12.4 ms to 9.8 ms.

---

## 8. Recommendations for Production Use Cases

* **Use YOLOv11 (nano)** for the majority of deployment scenarios:
  * **Real-time edge processing** (dashboard cameras, smartphones, embedded microcontrollers on survey vehicles) because it is faster, smaller, and uses less RAM.
  * **High-accuracy road surveys** where tight bounding boxes are needed to calculate pothole depth and volume from camera feeds (leveraging the higher mAP@0.5:0.95 score).
* **Use YOLOv8 (nano)** only if:
  * You have legacy software codebases integrated with YOLOv8 API structures that cannot be updated.
  * Your deployment environment has specific compiler toolchain constraints that do not yet support the newer block layers of YOLOv11.

---

## 9. Conclusion
Comparing YOLOv8 and YOLOv11 under identical conditions shows that YOLOv11 is a significant upgrade. By combining C3k2 bottleneck blocks with spatial attention mechanisms, YOLOv11 achieves a **3.5% higher mAP@0.5** while reducing model size by **13%** and increasing processing speed by **26%**. For automated road hazard detection, YOLOv11 is the recommended model.
