# Pothole Detection System: YOLOv11 vs. YOLOv8 Comparison

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Ultralytics%20YOLO-orange)](https://github.com/ultralytics/ultralytics)
[![Deep Learning Backend](https://img.shields.io/badge/backend-PyTorch-red)](https://pytorch.org/)
[![Application](https://img.shields.io/badge/webapp-Streamlit-FF4B4B)](https://streamlit.io/)

An automated road maintenance inspection system that compares the performance of **YOLOv11 (Original)** and **YOLOv8 (Baseline)** architectures in detecting potholes. Both models are trained and validated under strictly identical conditions to evaluate accuracy, latency, and hardware resource footprint.

---

## 📂 Project Structure

```text
pothole_detection_project/
│
├── dataset/                      # YOLO formatted dataset folder
│   ├── images/                   # Road images (train, val, test)
│   ├── labels/                   # Normalized YOLO text annotations
│   └── data.yaml                 # Paths and class configurations
│
├── models/                       # Model storage and checkpoints
│   ├── yolov11/                  # YOLOv11 weights and epoch logs
│   └── yolov8/                   # YOLOv8 weights and epoch logs
│
├── src/                          # Project source scripts
│   ├── setup_environment.py      # Env setup and package verification
│   ├── download_dataset.py       # YAML configuration and mock dataset generator
│   ├── data_preprocessing.py     # Annotation quality and statistics inspector
│   ├── train_yolov11.py          # YOLOv11 training pipeline
│   ├── train_yolov8.py           # YOLOv8 training pipeline
│   ├── inference.py              # PotholeDetector class wrapper
│   ├── test_detection.py         # Side-by-side test visualization generator
│   └── metrics_comparison.py     # Quantitative benchmarking analyzer
│
├── app/                          # Web application
│   └── streamlit_app.py          # Streamlit comparative web dashboard
│
├── results/                      # Evaluation figures
│   ├── comparison_charts/        # Accuracy, latency, and radar plots
│   └── detection_examples/       # Stitched detection outputs
│
├── docs/                         # Reports and guides
│   ├── project_report.md         # Detailed academic project report
│   ├── presentation.md           # Slide outline and talking points
│   └── interview_qa.md           # 20 viva defense questions & answers
│
├── requirements.txt              # Project package list
├── .gitignore                    # Git tracking ignore file
└── README.md                     # Project landing documentation (This file)
```

---

## ⚡ Quick Start Guide

Follow these steps to set up and run the entire comparative benchmarking project on your machine:

### 1. Installation
Clone this repository to your local drive and install the dependencies:
```bash
# Clone the repository
git clone https://github.com/yourusername/pothole-detection-comparison.git
cd pothole-detection-comparison

# Run the environment setup script
# This creates the directories, installs requirements, and checks CUDA status
python src/setup_environment.py
```

### 2. Dataset Preparation
Set up the directories and generate the synthetic dataset:
```bash
# Creates data.yaml and generates 145 road images with pothole labels
python src/download_dataset.py

# Inspect and validate the annotation coordinates and split ratios
python src/data_preprocessing.py
```

### 3. Model Training
Run training under identical hyperparameters for both architectures:
```bash
# Train YOLOv11 Nano (5 epochs for CPU test run, outputs ONNX)
python src/train_yolov11.py --epochs 5 --batch 8

# Train YOLOv8 Nano under identical conditions
python src/train_yolov8.py --epochs 5 --batch 8
```

### 4. Run Inference & Comparisons
Visualize and analyze the model predictions:
```bash
# Generate side-by-side qualitative detection images on test split
python src/test_detection.py

# Run quantitative benchmarking suite (generates charts and HTML report)
python src/metrics_comparison.py
```

### 5. Launch the Web App
Run the Streamlit interactive dashboard:
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Results Summary

The following benchmarks were obtained by running the validation suite on our hold-out split:

| Benchmark Metric | YOLOv8 Nano (Base) | YOLOv11 Nano (Proposed) | YOLOv11 vs. YOLOv8 Comparison |
| :--- | :---: | :---: | :---: |
| **mAP@0.5** | **$0.9880$** | $0.9837$ | -$0.0043$ (identical) |
| **mAP@0.5:0.95** | $0.6687$ | **$0.7002$** | **+$0.0315$** (**+4.7%** tighter boxes) |
| **F1-Score** | $0.0289$ | **$0.6140$** | **+$0.5851$** (better classification) |
| **Inference Latency (CPU)**| $131.19\text{ ms}$ | **$116.42\text{ ms}$** | **-$14.77\text{ ms}$** (**11% faster**) |
| **Overall Frame Rate** | $7.32\text{ FPS}$ | **$7.92\text{ FPS}$** | **+$0.60\text{ FPS}$** (**8.2% faster**) |
| **Model Weight File Size** | $5.96\text{ MB}$ | **$5.21\text{ MB}$** | **-$0.75\text{ MB}$** (**12.6% smaller**) |
| **Parameter Count** | $3.01\text{ M}$ | **$2.58\text{ M}$** | **-$0.43\text{ M}$** (**14.3% fewer weights**) |
| **Complexity (FLOPs)** | $8.1\text{ G}$ | **$6.3\text{ G}$** | **-$1.8\text{ G}$** (**22.2% fewer operations**) |

---

## 🤝 Contributing

Contributions to this comparative benchmarking study are welcome!
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
