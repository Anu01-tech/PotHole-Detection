import os
import sys
import time
import argparse
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

def print_separator():
    """Prints a styled separator line."""
    print("=" * 60)

def plot_custom_training_curves(csv_path, output_png_path):
    """
    Reads the ultralytics results.csv file, cleans column headers,
    and plots loss curves and accuracy (mAP) progression side-by-side.
    """
    if not os.path.exists(csv_path):
        print(f"  [WARNING] Results CSV file not found at: {csv_path}. Skipping curve plotting.")
        return
        
    try:
        # Load training log
        df = pd.read_csv(csv_path)
        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]
        
        # Check if epoch and key columns exist
        epoch_col = 'epoch'
        train_box = 'train/box_loss'
        train_cls = 'train/cls_loss'
        train_dfl = 'train/dfl_loss'
        val_box = 'val/box_loss'
        val_cls = 'val/cls_loss'
        val_dfl = 'val/dfl_loss'
        map50 = 'metrics/mAP50(B)'
        map95 = 'metrics/mAP50-95(B)'
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot Loss Curves
        axes[0].plot(df[epoch_col], df[train_box], label='Train Box Loss', color='#1f77b4', linestyle='-', linewidth=2)
        axes[0].plot(df[epoch_col], df[val_box], label='Val Box Loss', color='#aec7e8', linestyle='--', linewidth=2)
        axes[0].plot(df[epoch_col], df[train_cls], label='Train Class Loss', color='#ff7f0e', linestyle='-', linewidth=2)
        axes[0].plot(df[epoch_col], df[val_cls], label='Val Class Loss', color='#ffbb78', linestyle='--', linewidth=2)
        axes[0].plot(df[epoch_col], df[train_dfl], label='Train DFL Loss', color='#2ca02c', linestyle='-', linewidth=2)
        axes[0].plot(df[epoch_col], df[val_dfl], label='Val DFL Loss', color='#98df8a', linestyle='--', linewidth=2)
        axes[0].set_title("YOLOv8 Training and Validation Loss Curves", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Epoch", fontsize=10)
        axes[0].set_ylabel("Loss Value", fontsize=10)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(loc='upper right')
        
        # Plot mAP metrics progression
        axes[1].plot(df[epoch_col], df[map50], label='mAP@0.5', color='#2ca02c', linewidth=2.5)
        axes[1].plot(df[epoch_col], df[map95], label='mAP@0.5:0.95', color='#d62728', linewidth=2.5)
        axes[1].set_title("YOLOv8 Precision Progression (mAP)", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Epoch", fontsize=10)
        axes[1].set_ylabel("Metric Value (0 - 1)", fontsize=10)
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_png_path, dpi=150)
        plt.close()
        print(f"  [SUCCESS] Plot curves successfully saved to: {output_png_path}")
    except Exception as e:
        print(f"  [ERROR] Failed to generate custom training curves: {e}")

def main():
    # Parse command line arguments for quick verification runs
    parser = argparse.ArgumentParser(description="YOLOv8 Pothole Training Script")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    print_separator()
    print("          PHASE 3: YOLOv8 POTHOLE DETECTION TRAINING")
    print_separator()
    
    # Establish project directory paths
    data_yaml = os.path.abspath("dataset/data.yaml")
    results_dir = os.path.abspath("models/yolov8/results")
    weights_dir = os.path.abspath("models/yolov8/weights")
    
    if not os.path.exists(data_yaml):
        print(f"  [ERROR] Dataset configuration file not found at: {data_yaml}")
        print("  Please run 'python src/prepare_dataset.py' to generate the data splits first.")
        sys.exit(1)
        
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the official pre-trained YOLOv8 nano model
    print("  [INFO] Loading YOLOv8 nano weights for transfer learning...")
    model = YOLO("yolov8n.pt")
    
    # Define hyperparameter block
    training_params = {
        'data': data_yaml,
        'epochs': args.epochs,
        'imgsz': 640,
        'batch': args.batch,
        'patience': 10,
        'save': True,
        'device': 'auto',
        'workers': 4,              # Lowered for stable CPU/thread scaling
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'project': results_dir,
        'name': 'train_run',
        'exist_ok': True,
        'plots': True              # Autogenerate native ultralytics curves
    }
    
    # Resolve 'auto' device selection based on PyTorch's hardware detection to avoid crashes
    import torch
    if training_params['device'] == 'auto':
        training_params['device'] = 0 if torch.cuda.is_available() else 'cpu'
        print(f"  [INFO] Mapped device 'auto' -> '{training_params['device']}' (CUDA available: {torch.cuda.is_available()})")
        
    print("  [INFO] Hyperparameter Configuration:")
    for k, v in training_params.items():
        print(f"    - {k}: {v}")
        
    print_separator()
    print("  [INFO] Initiating YOLOv8 training loop...")
    print_separator()
    
    start_time = time.time()
    
    # Execute model training
    try:
        model.train(**training_params)
    except Exception as e:
        print(f"  [ERROR] An error occurred during YOLOv8 training: {e}")
        sys.exit(1)
        
    end_time = time.time()
    
    # Calculate formatted duration (minutes:seconds)
    duration_sec = end_time - start_time
    minutes, seconds = divmod(duration_sec, 60)
    time_str = f"{int(minutes)} minutes {int(seconds)} seconds"
    
    # Define checkpoint file locations
    best_pt_src = os.path.join(results_dir, "train_run", "weights", "best.pt")
    last_pt_src = os.path.join(results_dir, "train_run", "weights", "last.pt")
    
    best_pt_dest = os.path.join(weights_dir, "best.pt")
    last_pt_dest = os.path.join(weights_dir, "last.pt")
    
    # Copy best/last weights to their dedicated model directories
    if os.path.exists(best_pt_src):
        shutil.copy2(best_pt_src, best_pt_dest)
    if os.path.exists(last_pt_src):
        shutil.copy2(last_pt_src, last_pt_dest)
        
    # Read final loss values from training history log
    csv_path = os.path.join(results_dir, "train_run", "results.csv")
    final_train_loss = "N/A"
    final_val_loss = "N/A"
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if not df.empty:
                last_row = df.iloc[-1]
                # Sum box, class, and DFL losses to report a total training and val loss
                box_l = last_row.get('train/box_loss', 0.0)
                cls_l = last_row.get('train/cls_loss', 0.0)
                dfl_l = last_row.get('train/dfl_loss', 0.0)
                final_train_loss = f"{box_l + cls_l + dfl_l:.3f}"
                
                v_box_l = last_row.get('val/box_loss', 0.0)
                v_cls_l = last_row.get('val/cls_loss', 0.0)
                v_dfl_l = last_row.get('val/dfl_loss', 0.0)
                final_val_loss = f"{v_box_l + v_cls_l + v_dfl_l:.3f}"
        except Exception as e:
            print(f"  [WARNING] Could not parse final losses from CSV: {e}")
            
    # Plot custom charts and save them in results
    curve_output_path = os.path.join(results_dir, "yolov8_training_curves.png")
    plot_custom_training_curves(csv_path, curve_output_path)
    
    # Print final summary as requested
    print_separator()
    print("✅ YOLOv8 Training Complete")
    print(f"⏱️  Training Time: {time_str}")
    print(f"📊 Final Training Loss: {final_train_loss}")
    print(f"📊 Final Validation Loss: {final_val_loss}")
    print(f"💾 Model saved to: {best_pt_dest}")
    print_separator()

if __name__ == "__main__":
    main()
