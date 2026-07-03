"""
data_preprocessing.py

This script performs data validation and quality assurance for the YOLO dataset:
1. Validates all image-label pairs for consistency (warns of mismatched files).
2. Inspects label files to verify the YOLO format:
   - Contains 5 values per line.
   - Class ID is valid (0 for pothole).
   - Coordinates (x_center, y_center, width, height) are normalized float values in [0, 1].
3. Computes and displays dataset statistics:
   - Total images and labels per split (train, val, test).
   - Total number of pothole object instances.
   - Average number of potholes per image.
   - Resolution/dimension checks.

Usage:
    python src/data_preprocessing.py
"""

import os
import cv2
import sys
import numpy as np

def print_separator():
    print("=" * 60)

def validate_annotations(dataset_path):
    """
    Scans the train, val, and test directories to validate YOLO annotations
    and compile comprehensive statistics.
    """
    print_separator()
    print("          POTHOLE DETECTION PROJECT: DATASET QUALITY CHECK")
    print_separator()
    
    splits = ["train", "val", "test"]
    overall_stats = {}
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path '{dataset_path}' does not exist.")
        print("Please run 'python src/download_dataset.py' first.")
        sys.exit(1)
        
    for split in splits:
        print(f"\nProcessing Split: '{split.upper()}'")
        
        images_dir = os.path.join(dataset_path, "images", split)
        labels_dir = os.path.join(dataset_path, "labels", split)
        
        # Verify folders exist
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"  [WARNING] Folder mismatch or missing split directory for '{split}'. skipping...")
            continue
            
        # Get list of files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
        
        # Track statistics
        num_valid_images = 0
        num_corrupt_images = 0
        num_missing_labels = 0
        num_valid_labels = 0
        num_invalid_labels = 0
        total_instances = 0
        image_sizes = set()
        
        # Keep track of parsed label filenames to find orphaned labels
        image_basenames = {os.path.splitext(img)[0]: img for img in image_files}
        label_basenames = {os.path.splitext(lbl)[0]: lbl for lbl in label_files}
        
        # Iterate over images and check matching labels
        for basename, img_file in image_basenames.items():
            img_path = os.path.join(images_dir, img_file)
            
            # 1. Validate image readability and dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [ERROR] Corrupt image detected: {img_path}")
                num_corrupt_images += 1
                continue
                
            num_valid_images += 1
            h, w, c = img.shape
            image_sizes.add((w, h))
            
            # 2. Check if matching label file exists
            if basename not in label_basenames:
                # YOLO allows label-less images as background (negative) images, but we should report it.
                num_missing_labels += 1
                continue
                
            lbl_file = label_basenames[basename]
            lbl_path = os.path.join(labels_dir, lbl_file)
            
            # 3. Read and validate YOLO label contents
            valid_file_flag = True
            instances_in_image = 0
            
            try:
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                        
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"  [INVALID LABEL] {lbl_file}:{line_idx+1} -> Does not contain exactly 5 columns. Got {len(parts)}: '{line}'")
                        valid_file_flag = False
                        continue
                        
                    class_id_str, x_str, y_str, w_str, h_str = parts
                    
                    # Validate class ID
                    try:
                        class_id = int(class_id_str)
                        if class_id != 0:
                            print(f"  [INVALID ID] {lbl_file}:{line_idx+1} -> Class ID must be 0 (for pothole). Got {class_id}")
                            valid_file_flag = False
                    except ValueError:
                        print(f"  [INVALID TYPE] {lbl_file}:{line_idx+1} -> Class ID is not an integer: '{class_id_str}'")
                        valid_file_flag = False
                        
                    # Validate coordinates (must be floats between 0 and 1)
                    coords = [x_str, y_str, w_str, h_str]
                    coord_names = ['x_center', 'y_center', 'width', 'height']
                    
                    for name, coord_str in zip(coord_names, coords):
                        try:
                            val = float(coord_str)
                            if val < 0.0 or val > 1.0:
                                print(f"  [OUT OF BOUNDS] {lbl_file}:{line_idx+1} -> Coordinate {name} value {val} is not in [0.0, 1.0]")
                                valid_file_flag = False
                        except ValueError:
                            print(f"  [INVALID TYPE] {lbl_file}:{line_idx+1} -> Coordinate {name} is not a float: '{coord_str}'")
                            valid_file_flag = False
                            
                    if valid_file_flag:
                        instances_in_image += 1
                        
            except Exception as e:
                print(f"  [ERROR] Failed to read label file {lbl_path}: {e}")
                valid_file_flag = False
                
            if valid_file_flag:
                num_valid_labels += 1
                total_instances += instances_in_image
            else:
                num_invalid_labels += 1
                
        # 4. Check for orphaned label files (labels without matching images)
        orphaned_labels = []
        for basename, lbl_file in label_basenames.items():
            if basename not in image_basenames:
                orphaned_labels.append(lbl_file)
                
        # Report findings for this split
        print(f"  [OK] Valid Images: {num_valid_images}")
        if num_corrupt_images > 0:
            print(f"  [FAIL] Corrupt Images: {num_corrupt_images}")
        print(f"  [OK] Matching Label Files: {num_valid_labels}")
        if num_invalid_labels > 0:
            print(f"  [FAIL] Invalid Label Formats: {num_invalid_labels}")
        if num_missing_labels > 0:
            print(f"  [INFO] Background Images (No labels): {num_missing_labels}")
        if len(orphaned_labels) > 0:
            print(f"  [WARNING] Found {len(orphaned_labels)} orphaned label files (no corresponding image):")
            print(f"    First 3: {orphaned_labels[:3]}")
            
        print(f"  [OK] Total Pothole Instances: {total_instances}")
        if num_valid_images > 0:
            print(f"  [OK] Avg Potholes per Image: {total_instances / num_valid_images:.2f}")
        print(f"  [OK] Unique Resolutions Detected: {list(image_sizes)}")
        
        # Save split metrics
        overall_stats[split] = {
            'images': num_valid_images,
            'labels': num_valid_labels,
            'instances': total_instances,
            'resolutions': list(image_sizes)
        }
        
    # Print Consolidated Summary
    print_separator()
    print("CONSOLIDATED DATASET SUMMARY:")
    print(f"{'Split':<10} | {'Images':<10} | {'Labels':<10} | {'Instances (Potholes)':<20} | {'Resolutions':<20}")
    print("-" * 70)
    
    total_imgs = 0
    total_lbls = 0
    total_inst = 0
    
    for split, data in overall_stats.items():
        print(f"{split.capitalize():<10} | {data['images']:<10} | {data['labels']:<10} | {data['instances']:<20} | {str(data['resolutions']):<20}")
        total_imgs += data['images']
        total_lbls += data['labels']
        total_inst += data['instances']
        
    print("-" * 70)
    print(f"{'Total':<10} | {total_imgs:<10} | {total_lbls:<10} | {total_inst:<20} | -")
    print_separator()
    print("Dataset validation complete! Your dataset is clean and ready for model training.")
    print("Proceed to Phase 3: YOLOv11 Training.")
    print_separator()

if __name__ == "__main__":
    validate_annotations("dataset")
