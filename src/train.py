import os
import sys
import time
import shutil
import torch
import pandas as pd
from ultralytics import YOLO

# Fix Windows console unicode issues
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def main():
    print("🚀 Starting YOLOv8 Training Setup...")
    
    # 1. Path definitions
    data_yaml = "dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset configuration not found at {data_yaml}! Please run prepare_dataset.py first.")
        sys.exit(1)
        
    # Define destination folders
    weights_dir = "models/yolov8/weights"
    results_dir = "models/yolov8/results"
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Load pre-trained YOLOv8 Nano model
    print("📥 Loading pre-trained YOLOv8n weights...")
    model = YOLO('yolov8n.pt')

    # Determine training device (GPU if available, otherwise CPU)
    # This prevents YOLO from failing with device='auto' on CPU-only machines
    if torch.cuda.is_available():
        device = '0'
        print(f"✅ CUDA GPU detected! Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("ℹ️  No CUDA GPU detected. Training will run on CPU (this might be slow).")

    # Record training start time
    start_time = time.time()
    
    # 3. Train the model
    # Hyperparameters match the requested configuration
    print(f"🏃 Training model on device: {device}...")
    train_results = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=320,
        batch=16,
        patience=10,        # Stop training if mAP does not improve for 10 epochs
        save=True,
        device=device,      # Use 'cpu' or GPU device index
        workers=0,          # Set workers=0 on Windows to avoid multiprocessing/freeze_support bugs
        pretrained=True,    # Starts training from pre-trained COCO weights (transfer learning)
        optimizer='AdamW',  # Adam with weight decay, highly robust optimizer
        lr0=0.001,          # Initial learning rate
        lrf=0.01,           # Final learning rate fraction (lr0 * lrf)
        momentum=0.937,     # Optimizer momentum
        weight_decay=0.0005,# Weight decay for regularization (prevents overfitting)
        warmup_epochs=3.0,  # Number of epochs to warm up learning rate from 0
        project=os.path.abspath('models/yolov8/results'),  # Root directory for saving results
        name='train',       # Sub-folder name under project root
        exist_ok=True       # Overwrite/use existing folder without throwing error
    )
    
    # Record training end time
    end_time = time.time()
    duration_sec = end_time - start_time
    
    # Format duration
    hours, remainder = divmod(int(duration_sec), 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = ""
    if hours > 0:
        time_str += f"{hours} hours "
    if minutes > 0:
        time_str += f"{minutes} minutes "
    time_str += f"{seconds} seconds"

    print("\n✅ Training run finished. Processing metrics...")

    # 4. Copy weights to the requested project structure locations
    run_weights_dir = os.path.join(results_dir, "train", "weights")
    best_src = os.path.join(run_weights_dir, "best.pt")
    last_src = os.path.join(run_weights_dir, "last.pt")
    
    best_dst = os.path.join(weights_dir, "best.pt")
    last_dst = os.path.join(weights_dir, "last.pt")
    
    # Verify and copy files
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"💾 Copied best weights to: {best_dst}")
    else:
        print(f"⚠️  Best weights file not found at {best_src}")
        
    if os.path.exists(last_src):
        shutil.copy2(last_src, last_dst)
        print(f"💾 Copied last weights to: {last_dst}")
    else:
        print(f"⚠️  Last weights file not found at {last_src}")
        
    # 5. Copy training curve plot
    curves_src = os.path.join(results_dir, "train", "results.png")
    curves_dst = os.path.join(results_dir, "training_curves.png")
    if os.path.exists(curves_src):
        shutil.copy2(curves_src, curves_dst)
        print(f"📈 Copied training curves plot to: {curves_dst}")
    else:
        print(f"⚠️  Training curves plot not found at {curves_src}")

    # 6. Parse results.csv to extract final statistics
    csv_path = os.path.join(results_dir, "train", "results.csv")
    final_train_loss = "N/A"
    final_val_loss = "N/A"
    best_map50 = "N/A"
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Remove whitespace from column names
            df.columns = [c.strip() for c in df.columns]
            
            # Read last row for final epoch results
            last_row = df.iloc[-1]
            
            # Calculate sum of losses
            train_box = last_row.get('train/box_loss', 0)
            train_cls = last_row.get('train/cls_loss', 0)
            train_dfl = last_row.get('train/dfl_loss', 0)
            final_train_loss = f"{train_box + train_cls + train_dfl:.4f}"
            
            val_box = last_row.get('val/box_loss', 0)
            val_cls = last_row.get('val/cls_loss', 0)
            val_dfl = last_row.get('val/dfl_loss', 0)
            final_val_loss = f"{val_box + val_cls + val_dfl:.4f}"
            
            # Get best mAP@0.5 from the entire history
            best_map50 = f"{df['metrics/mAP50(B)'].max():.4f}"
        except Exception as e:
            print(f"⚠️  Error parsing results.csv: {e}")
            
    # 7. Print final training summary in the requested format
    print("\n" + "=" * 40)
    print("🚀 Starting YOLOv8 Training...")
    print(f"⏱️  Training Time: {time_str}")
    print(f"📊 Final Training Loss: {final_train_loss}")
    print(f"📊 Final Validation Loss: {final_val_loss}")
    print(f"📈 Best mAP@0.5: {best_map50}")
    print(f"💾 Model saved: {best_dst}")
    print("✅ Training Complete!")
    print("=" * 40)

if __name__ == "__main__":
    main()
