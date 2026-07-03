"""
train_yolov8.py

This script trains a YOLOv8 (nano) model for pothole detection under identical conditions to YOLOv11.
It includes features to:
1. Load pre-trained YOLOv8 weights (yolov8n.pt) for Transfer Learning.
2. Configure all hyperparameters identically to YOLOv11 (learning rate, augmentations, optimizer).
3. Support training resume from a checkpoint.
4. Measure and log total training time.
5. Save the best and last checkpoints to 'models/yolov8/weights/'.
6. Export the finalized best model to ONNX format.

Usage:
    # Standard training (50 epochs):
    python src/train_yolov8.py --epochs 50 --batch 16
    
    # Fast test run (1 epoch, CPU training):
    python src/train_yolov8.py --epochs 1 --batch 8
    
    # Resume training:
    python src/train_yolov8.py --resume
"""

import os
import sys
import time
import shutil
import argparse
from ultralytics import YOLO

def print_separator():
    print("=" * 60)

def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="YOLOv8 Pothole Detection Training Script")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img", type=int, default=640, help="Image size")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--export", action="store_true", default=True, help="Export model to ONNX after training")
    args = parser.parse_args()
    
    print_separator()
    print("          POTHOLE DETECTION PROJECT: YOLOv8 TRAINING")
    print_separator()
    
    # Define paths (Structured for YOLOv8)
    dataset_yaml = os.path.abspath("dataset/data.yaml")
    results_dir = os.path.abspath("models/yolov8/results")
    weights_dir = os.path.abspath("models/yolov8/weights")
    
    # Verify dataset configuration exists
    if not os.path.exists(dataset_yaml):
        print(f"[ERROR] dataset config file not found at: {dataset_yaml}")
        print("Please run 'python src/download_dataset.py' to generate the dataset first.")
        sys.exit(1)
        
    os.makedirs(weights_dir, exist_ok=True)
    
    # 2. Model Initialization (Transfer Learning)
    if args.resume:
        last_checkpoint = os.path.join(results_dir, "train_run", "weights", "last.pt")
        if not os.path.exists(last_checkpoint):
            print(f"[ERROR] Last checkpoint not found at: {last_checkpoint}")
            print("Cannot resume training without a valid checkpoint.")
            sys.exit(1)
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        model = YOLO(last_checkpoint)
    else:
        print("Initializing YOLOv8 (nano) model with pre-trained weights for Transfer Learning...")
        # Ultralytics will download 'yolov8n.pt' automatically if not present locally
        model = YOLO("yolov8n.pt")
        
    # Print model architecture info
    print(f"Model Class: {model.__class__.__name__}")
    print(f"Task: {model.task}")
    
    # 3. Train Model (Parameters are identical to YOLOv11 for fair comparison)
    print("\nStarting training loop with identical hyperparameters...")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch}")
    print(f"  - Image Resolution: {args.img}x{args.img}")
    
    start_time = time.time()
    
    try:
        # We call the train method of the model and pass all hyperparameters explicitly
        results = model.train(
            data=dataset_yaml,            # Path to data.yaml
            epochs=args.epochs,           # Number of epochs to train for
            batch=args.batch,             # Batch size
            imgsz=args.img,               # Image resolution
            
            # Optimization Parameters (Identical)
            optimizer='AdamW',            # AdamW optimizer with weight decay
            lr0=0.001,                    # Initial learning rate
            lrf=0.01,                     # Final learning rate fraction (lr0 * lrf)
            momentum=0.937,               # Momentum
            weight_decay=0.0005,          # Weight decay
            
            # Training Settings (Identical)
            patience=10,                  # Early stopping patience
            save=True,                    # Save checkpoints
            resume=args.resume,           # Resume training if flag is True
            device=None,                  # Auto-detects device (uses CUDA GPU if available, else CPU)
            
            # Folder Outputs (Structured for YOLOv8)
            project=results_dir,          # Output directory project folder (absolute path)
            name="train_run",             # Output subfolder name
            exist_ok=True,                # Overwrite existing folders
            
            # Data Augmentation Parameters (Identical to YOLOv11)
            hsv_h=0.015,                  # HSV-Hue augmentation (fraction)
            hsv_s=0.7,                    # HSV-Saturation augmentation (fraction)
            hsv_v=0.4,                    # HSV-Value augmentation (fraction)
            degrees=10.0,                 # Image rotation (degrees)
            translate=0.1,                # Image translation (fraction)
            scale=0.5,                    # Image scale (gain)
            shear=0.0,                    # Image shear (degrees)
            perspective=0.0,              # Image perspective (fraction)
            flipud=0.0,                   # Vertical flip probability
            fliplr=0.5,                   # Horizontal flip probability
            mosaic=1.0,                   # Mosaic augmentation probability (joins 4 images)
            mixup=0.0,                    # Mixup augmentation probability (overlays 2 images)
            copy_paste=0.0,               # Copy-Paste augmentation probability
            
            # Miscellaneous
            plots=True                    # Generate and save metrics plots (loss curves, precision-recall)
        )
        
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        hours, rem = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        
        print_separator()
        print(f"TRAINING COMPLETED SUCCESSFULLY in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}!")
        print_separator()
        
        # 4. Copy best and last checkpoints to the desired project folders
        best_pt_src = os.path.join(results_dir, "train_run", "weights", "best.pt")
        last_pt_src = os.path.join(results_dir, "train_run", "weights", "last.pt")
        
        best_pt_dest = os.path.join(weights_dir, "best.pt")
        last_pt_dest = os.path.join(weights_dir, "last.pt")
        
        if os.path.exists(best_pt_src):
            shutil.copy(best_pt_src, best_pt_dest)
            print(f"  + Saved best model weights to: {best_pt_dest}")
        if os.path.exists(last_pt_src):
            shutil.copy(last_pt_src, last_pt_dest)
            print(f"  + Saved last model weights to: {last_pt_dest}")
            
        # 5. Export Best Model to ONNX format
        if args.export and os.path.exists(best_pt_dest):
            print("\nExporting best model to ONNX format...")
            best_model = YOLO(best_pt_dest)
            onnx_path = best_model.export(format="onnx")
            
            # Copy exported ONNX file to weights directory
            onnx_src = os.path.join(results_dir, "train_run", "weights", "best.onnx")
            onnx_dest = os.path.join(weights_dir, "best.onnx")
            
            if os.path.exists(onnx_src) and os.path.abspath(onnx_src) != os.path.abspath(onnx_dest):
                shutil.copy(onnx_src, onnx_dest)
                print(f"  + Saved best model ONNX file to: {onnx_dest}")
            elif os.path.exists(onnx_path) and os.path.abspath(onnx_path) != os.path.abspath(onnx_dest):
                shutil.copy(onnx_path, onnx_dest)
                print(f"  + Saved best model ONNX file to: {onnx_dest}")
            else:
                print(f"  + Saved best model ONNX file directly: {onnx_dest}")
                
        print("\nAll checkpoints and logs successfully stored!")
        print("Proceed to Phase 5: Inference and Visualization.")
        print_separator()
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {e}")
        print_separator()
        sys.exit(1)

if __name__ == "__main__":
    main()
