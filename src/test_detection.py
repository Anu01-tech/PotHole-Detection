"""
test_detection.py

This script performs side-by-side detection comparisons between YOLOv8 and YOLOv11:
1. Loads the best weights for both YOLOv8 and YOLOv11.
2. Selects sample test images from 'dataset/images/test/'.
3. Runs inference on each image using both models.
4. Combines the output images horizontally side-by-side.
5. Adds titles ("YOLOv8" vs "YOLOv11") on the images for readability.
6. Saves the side-by-side comparison files in 'results/detection_examples/'.

Usage:
    python src/test_detection.py
"""

import os
import cv2
import sys
import numpy as np
from inference import PotholeDetector

def print_separator():
    print("=" * 60)

def main():
    print_separator()
    print("          POTHOLE DETECTION PROJECT: VISUAL COMPARISON")
    print_separator()
    
    # Define paths
    test_images_dir = "dataset/images/test"
    output_dir = "results/detection_examples"
    
    yolov8_model_path = "models/yolov8/weights/best.pt"
    yolov11_model_path = "models/yolov11/weights/best.pt"
    
    # 1. Verify model weights exist
    if not os.path.exists(yolov8_model_path) or not os.path.exists(yolov11_model_path):
        print("[ERROR] Model weights not found.")
        print(f"  YOLOv8 weights path: {yolov8_model_path}")
        print(f"  YOLOv11 weights path: {yolov11_model_path}")
        print("Please ensure both training scripts have been run first.")
        sys.exit(1)
        
    # 2. Get list of test images
    if not os.path.exists(test_images_dir):
        print(f"[ERROR] Test images folder '{test_images_dir}' not found.")
        sys.exit(1)
        
    image_exts = (".jpg", ".jpeg", ".png")
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(image_exts)]
    
    if not test_images:
        print(f"[WARNING] No test images found in {test_images_dir}.")
        sys.exit(0)
        
    print(f"Found {len(test_images)} test images. Setting up detectors...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Initialize detectors
    try:
        detector_v8 = PotholeDetector(yolov8_model_path)
        detector_v11 = PotholeDetector(yolov11_model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load detectors: {e}")
        sys.exit(1)
        
    print("\nRunning inference and building side-by-side comparison images...")
    
    # We will compare up to 5 images for visualization
    max_images_to_compare = min(5, len(test_images))
    
    for i in range(max_images_to_compare):
        img_name = test_images[i]
        img_path = os.path.join(test_images_dir, img_name)
        
        print(f"  Processing image [{i+1}/{max_images_to_compare}]: {img_name}")
        
        # Run inference using both models
        # Using a confidence threshold of 0.01 and NMS IoU threshold of 0.45
        annotated_v8, detections_v8 = detector_v8.predict_image(img_path, conf_threshold=0.01, iou_threshold=0.45)
        annotated_v11, detections_v11 = detector_v11.predict_image(img_path, conf_threshold=0.01, iou_threshold=0.45)
        
        # 4. Create Side-by-Side Canvas
        # Verify sizes are identical before stacking
        h, w, c = annotated_v8.shape
        if annotated_v11.shape != annotated_v8.shape:
            # Resize YOLOv11 to match YOLOv8 just in case
            annotated_v11 = cv2.resize(annotated_v11, (w, h))
            
        # Place them side-by-side: YOLOv8 on the left, YOLOv11 on the right
        comparison_canvas = np.zeros((h + 50, w * 2 + 10, 3), dtype=np.uint8) # Extra height at top for header, black separator line
        
        # Fill canvas with black background first
        comparison_canvas[:] = (20, 20, 20) # Sleek charcoal black color
        
        # Paste annotated images
        # Left side: YOLOv8 (shifted down by 50px header)
        comparison_canvas[50:, :w] = annotated_v8
        # Right side: YOLOv11 (shifted down by 50px header, shifted right by w + 10px separator width)
        comparison_canvas[50:, w + 10:] = annotated_v11
        
        # 5. Draw Header Titles
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        color_white = (255, 255, 255)
        thickness = 1
        
        # Header text for YOLOv8
        title_v8 = f"YOLOv8 Nano (Potholes: {len(detections_v8)})"
        (t_w_v8, t_h_v8), _ = cv2.getTextSize(title_v8, font, font_scale, thickness)
        pos_x_v8 = (w - t_w_v8) // 2
        cv2.putText(comparison_canvas, title_v8, (pos_x_v8, 32), font, font_scale, color_white, thickness, cv2.LINE_AA)
        
        # Header text for YOLOv11
        title_v11 = f"YOLOv11 Nano (Potholes: {len(detections_v11)})"
        (t_w_v11, t_h_v11), _ = cv2.getTextSize(title_v11, font, font_scale, thickness)
        pos_x_v11 = w + 10 + (w - t_w_v11) // 2
        cv2.putText(comparison_canvas, title_v11, (pos_x_v11, 32), font, font_scale, color_white, thickness, cv2.LINE_AA)
        
        # Print info in console
        print(f"    - YOLOv8 detected  : {len(detections_v8)} potholes")
        print(f"    - YOLOv11 detected : {len(detections_v11)} potholes")
        
        # Save comparison image
        output_name = f"comparison_{os.path.splitext(img_name)[0]}.jpg"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, comparison_canvas)
        print(f"    + Saved comparison plot to: {output_path}")
        
    print_separator()
    print("VISUAL COMPARISONS COMPLETED SUCCESSFULLY!")
    print(f"Check results in '{output_dir}/' folder.")
    print("Proceed to Phase 6: Metrics Comparison and Report Generation.")
    print_separator()

if __name__ == "__main__":
    main()
