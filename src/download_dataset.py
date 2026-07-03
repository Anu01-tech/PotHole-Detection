"""
download_dataset.py

This script prepares the dataset for the Pothole Detection system:
1. Creates the YOLO-compatible folder structure.
2. Generates a 'data.yaml' file that tells YOLO where to find the data and classes.
3. Automatically generates a synthetic/mock pothole dataset (textured road backgrounds 
   with dark ellipses representing potholes and corresponding YOLO labels) to ensure 
   the training scripts are runnable out-of-the-box.
4. Outputs detailed instructions for downloading and using the real Kaggle dataset.

Usage:
    python src/download_dataset.py
"""

import os
import sys
import yaml
import numpy as np
import cv2
import random

def print_separator():
    print("=" * 60)

def create_yolo_yaml(dataset_path):
    """
    Creates the data.yaml file required by YOLO.
    It contains:
    - path: Absolute path to the dataset directory.
    - train, val, test: Relative paths to split image folders.
    - names: Class index to class name mapping.
    """
    print("\n[1/3] Creating data.yaml configuration...")
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    # Standardize path with forward slashes for Windows compatibility
    abs_dataset_path = os.path.abspath(dataset_path).replace("\\", "/")
    
    yaml_data = {
        'path': abs_dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'pothole'
        }
    }
    
    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        print(f"  + Created: {yaml_path}")
        print(f"  - Configured absolute dataset path: {abs_dataset_path}")
    except Exception as e:
        print(f"  [ERROR] Failed to write data.yaml: {e}")
        sys.exit(1)

def generate_synthetic_potholes(dataset_path, split, count):
    """
    Generates synthetic road images and YOLO labels.
    - Road surface: Grey textured image with random noise.
    - Potholes: Randomly drawn dark, deformed ellipses on the road.
    - Annotations: Bounding box coordinates formatted in YOLO format (class_id x_center y_center width height)
    """
    print(f"  Generating {count} synthetic images for '{split}' split...")
    
    images_dir = os.path.join(dataset_path, "images", split)
    labels_dir = os.path.join(dataset_path, "labels", split)
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for i in range(count):
        # 1. Create a road-like textured background (640x640 pixels)
        # Base grey road color (typically RGB/BGR around 100-130)
        base_gray = random.randint(100, 140)
        img = np.full((640, 640, 3), base_gray, dtype=np.uint8)
        
        # Add random pixel-level noise to simulate asphalt texture
        noise = np.random.randint(-20, 20, (640, 640, 3))
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Optionally add crack lines or lane markings
        if random.random() > 0.5:
            # Crack line
            pts = np.array([[random.randint(0, 640), random.randint(0, 640)] for _ in range(5)], np.int32)
            cv2.polylines(img, [pts], False, (random.randint(60, 90), random.randint(60, 90), random.randint(60, 90)), random.randint(1, 2))
            
        # Draw potholes (between 1 and 4 per image)
        num_potholes = random.randint(1, 4)
        annotations = []
        
        for _ in range(num_potholes):
            # Random size and location
            x_center = random.randint(80, 560)
            y_center = random.randint(80, 560)
            axes_w = random.randint(30, 90)
            axes_h = random.randint(15, 45) # Ellipses are wider than tall generally
            
            # Dark color for pothole interior (simulating depth)
            pothole_color = (random.randint(30, 60), random.randint(30, 60), random.randint(30, 60))
            
            # Rotation of pothole
            angle = random.randint(-30, 30)
            
            # Draw the filled ellipse (depth)
            cv2.ellipse(img, (x_center, y_center), (axes_w, axes_h), angle, 0, 360, pothole_color, -1)
            
            # Draw a slightly lighter rim/border to simulate erosion/edge (depth effect)
            cv2.ellipse(img, (x_center, y_center), (axes_w, axes_h), angle, 0, 360, (pothole_color[0]+15, pothole_color[1]+15, pothole_color[2]+15), random.randint(1, 3))
            
            # Compute YOLO bounding box coordinates
            # Since the ellipse is rotated, we approximate the bounding box containing the ellipse
            w = int(2 * max(axes_w, axes_h))
            h = int(2 * max(axes_w, axes_h))
            
            # Keep bounding boxes within bounds
            x_min = max(0, x_center - w // 2)
            y_min = max(0, y_center - h // 2)
            x_max = min(640, x_center + w // 2)
            y_max = min(640, y_center + h // 2)
            
            # Recompute center, width, height from clamped values
            box_w = x_max - x_min
            box_h = y_max - y_min
            box_x = x_min + box_w / 2
            box_y = y_min + box_h / 2
            
            # Normalize coordinates to [0, 1] relative to image dimensions (640x640)
            norm_x = box_x / 640.0
            norm_y = box_y / 640.0
            norm_w = box_w / 640.0
            norm_h = box_h / 640.0
            
            # Class ID for pothole is 0
            annotations.append(f"0 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        # Save image file
        img_filename = f"road_{split}_{i:03d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        cv2.imwrite(img_path, img)
        
        # Save label file (must match image filename exactly, with .txt extension)
        lbl_filename = f"road_{split}_{i:03d}.txt"
        lbl_path = os.path.join(labels_dir, lbl_filename)
        with open(lbl_path, 'w') as f:
            f.write("\n".join(annotations))
            
    print(f"  + Generated {count} images & labels in dataset/images/{split} and dataset/labels/{split}")

def generate_dataset_suite(dataset_path):
    """Generates the train, validation, and test splits."""
    print("\n[2/3] Generating Synthetic YOLO Dataset (to enable instant running)...")
    # Generates a quick mock dataset
    generate_synthetic_potholes(dataset_path, "train", 100)
    generate_synthetic_potholes(dataset_path, "val", 30)
    generate_synthetic_potholes(dataset_path, "test", 15)
    print("  [SUCCESS] Synthetic dataset generated successfully!")

def print_kaggle_instructions():
    """Prints instructions on how the user can download and use the real dataset."""
    print_separator()
    print("KAGGLE DATASET DOWNLOAD & CUSTOMIZATION INSTRUCTIONS:")
    print("If you wish to use the real road pothole dataset:")
    print("1. Visit the Kaggle URL: https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset")
    print("2. Download the ZIP file containing 'Normal' and 'Potholes' images.")
    print("3. Since that dataset is for binary classification (it doesn't have bounding box files),")
    print("   you can annotate the images yourself using free tools like LabelImg, Roboflow, or CVAT.")
    print("   Or you can download a pre-annotated YOLO Pothole Dataset from Roboflow Universe:")
    print("   e.g. Search 'Pothole Detection YOLO' on Roboflow Universe (https://universe.roboflow.com)")
    print("4. Place your images in 'dataset/images/train', 'dataset/images/val', etc.")
    print("   and matching YOLO label files in 'dataset/labels/train', 'dataset/labels/val', etc.")
    print_separator()

def main():
    print_separator()
    print("          POTHOLE DETECTION PROJECT: DATASET PREPARATION")
    print_separator()
    
    dataset_path = "dataset"
    
    # 1. Create data.yaml
    create_yolo_yaml(dataset_path)
    
    # 2. Generate Synthetic Dataset
    generate_dataset_suite(dataset_path)
    
    # 3. Print Instructions
    print_kaggle_instructions()
    
    print("\n[3/3] DATASET SETUP COMPLETE!")
    print("You can now run 'python src/data_preprocessing.py' to validate the annotations!")
    print_separator()

if __name__ == "__main__":
    main()
